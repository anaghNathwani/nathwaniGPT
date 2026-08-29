"""
Load HuggingFace safetensors weights into our NathwaniGPT model.

Handles two common weight layouts:
  - Fused:    qkv_proj (Q+K+V combined) and gate_up_proj (gate+up combined)
  - Separate: q_proj / k_proj / v_proj / gate_proj / up_proj

Supports sharded checkpoints (model.safetensors.index.json) and
single-file checkpoints (model.safetensors).
"""

import json
from pathlib import Path
from typing import Union

import torch
from safetensors.torch import load_file

from .model import ModelConfig, NathwaniGPT


# ---------------------------------------------------------------------------
# Simple 1-to-1 mappings (global tensors + per-layer non-fused weights)
# ---------------------------------------------------------------------------

_GLOBAL_MAP = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight":         "norm.weight",
    "lm_head.weight":            "lm_head.weight",
}

_LAYER_PAIRS = [
    ("self_attn.o_proj.weight",           "self_attn.o_proj.weight"),
    ("mlp.down_proj.weight",              "mlp.down_proj.weight"),
    ("input_layernorm.weight",            "input_layernorm.weight"),
    ("post_attention_layernorm.weight",   "post_attention_layernorm.weight"),
    # separate-proj layout (Mistral, Llama, etc.)
    ("self_attn.q_proj.weight",           "self_attn.q_proj.weight"),
    ("self_attn.k_proj.weight",           "self_attn.k_proj.weight"),
    ("self_attn.v_proj.weight",           "self_attn.v_proj.weight"),
    ("mlp.gate_proj.weight",              "mlp.gate_proj.weight"),
    ("mlp.up_proj.weight",                "mlp.up_proj.weight"),
]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    weights_dir: Union[str, Path],
    device: str = "auto",
    dtype: torch.dtype = None,  # auto-selected per device
) -> tuple[NathwaniGPT, ModelConfig, str]:
    """
    Load a NathwaniGPT model from a HuggingFace weights directory.

    Returns (model, config, device).
    """
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_dir}.\n"
            f"Run: python scripts/download.py phi4-mini"
        )

    # Parse HF config — use actual values, no hardcoded fallbacks that might be wrong
    with open(weights_dir / "config.json") as f:
        hf_cfg = json.load(f)

    cfg = ModelConfig.from_dict({
        "vocab_size":              hf_cfg["vocab_size"],
        "hidden_size":             hf_cfg["hidden_size"],
        "intermediate_size":       hf_cfg["intermediate_size"],
        "num_hidden_layers":       hf_cfg["num_hidden_layers"],
        "num_attention_heads":     hf_cfg["num_attention_heads"],
        "num_key_value_heads":     hf_cfg.get("num_key_value_heads", hf_cfg["num_attention_heads"]),
        "max_position_embeddings": hf_cfg["max_position_embeddings"],
        "rms_norm_eps":            hf_cfg.get("rms_norm_eps", 1e-5),
        "rope_theta":              hf_cfg.get("rope_theta", 10000.0),
        "tie_word_embeddings":     hf_cfg.get("tie_word_embeddings", False),
    })

    if device == "auto":
        device = _pick_device()

    if dtype is None:
        # bfloat16 has known numerical issues on MPS for long sequences; use float16 there
        if device == "mps":
            dtype = torch.float16
        elif device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

    model = NathwaniGPT(cfg).to(dtype).to(device)
    print(f"[loader] Model: {cfg.num_hidden_layers}L · {cfg.hidden_size}H · "
          f"{cfg.num_attention_heads}A/{cfg.num_key_value_heads}KV · "
          f"{model.n_params()/1e9:.1f}B params · {device}")

    # Load safetensors (sharded or single file)
    index_path = weights_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        raw: dict[str, torch.Tensor] = {}
        for shard in shards:
            raw.update(load_file(weights_dir / shard))
    else:
        raw = load_file(weights_dir / "model.safetensors")

    our_state: dict[str, torch.Tensor] = {}

    # 1. Global 1-to-1 mappings
    for hf_name, our_name in _GLOBAL_MAP.items():
        if hf_name in raw:
            our_state[our_name] = raw[hf_name].to(dtype)

    # 2. Tied embeddings: lm_head shares embed_tokens weight
    if cfg.tie_word_embeddings and "lm_head.weight" not in our_state:
        our_state["lm_head.weight"] = our_state["embed_tokens.weight"]

    # 3. Per-layer weights
    q_dim  = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim

    for i in range(cfg.num_hidden_layers):
        hf_pre = f"model.layers.{i}."
        us_pre = f"layers.{i}."

        # Simple 1-to-1 per-layer weights
        for hf_suffix, our_suffix in _LAYER_PAIRS:
            key = hf_pre + hf_suffix
            if key in raw:
                our_state[us_pre + our_suffix] = raw[key].to(dtype)

        # Fused QKV → split into q / k / v
        qkv_key = hf_pre + "self_attn.qkv_proj.weight"
        if qkv_key in raw:
            qkv = raw[qkv_key].to(dtype)
            our_state[us_pre + "self_attn.q_proj.weight"] = qkv[:q_dim].contiguous()
            our_state[us_pre + "self_attn.k_proj.weight"] = qkv[q_dim : q_dim + kv_dim].contiguous()
            our_state[us_pre + "self_attn.v_proj.weight"] = qkv[q_dim + kv_dim :].contiguous()

        # Fused gate+up → split into gate / up
        gate_up_key = hf_pre + "mlp.gate_up_proj.weight"
        if gate_up_key in raw:
            gate_up = raw[gate_up_key].to(dtype)
            mid = gate_up.shape[0] // 2
            our_state[us_pre + "mlp.gate_proj.weight"] = gate_up[:mid].contiguous()
            our_state[us_pre + "mlp.up_proj.weight"]   = gate_up[mid:].contiguous()

    # Dequantize int8 weights produced by scripts/quantize.py
    quant_meta_path = weights_dir / "quantization.json"
    if quant_meta_path.exists():
        import json as _json
        scales = _json.loads(quant_meta_path.read_text())
        dequantized = 0
        for key in list(our_state.keys()):
            if key in scales and our_state[key].dtype == torch.int8:
                scale = torch.tensor(scales[key], dtype=torch.float32)
                our_state[key] = our_state[key].to(torch.float32).mul_(scale).to(dtype)
                dequantized += 1
        if dequantized:
            print(f"[loader] Dequantized {dequantized} int8 tensors → {dtype}")

    model.load_state_dict(our_state, strict=True)
    model.eval()
    print("[loader] All weights loaded.")
    return model, cfg, device
