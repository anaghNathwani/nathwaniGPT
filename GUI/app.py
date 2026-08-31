#!/usr/bin/env python3
"""
nathwaniGPT TUI launcher.

Starts the inference API server (serve/api.py) then opens the React frontend.

Dev mode  (default): spins up the Vite dev server alongside the API.
Prod mode (--prod):  serves the pre-built dist/ from FastAPI at the same port.

Usage:
    python tui/app.py
    python tui/app.py --weights weights/phi4-mini --api-port 8080
    python tui/app.py --prod      # after: cd tui/frontend && npm run build
"""
import argparse
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _start_api(weights: str, host: str, port: int) -> None:
    """Run the FastAPI server (blocks — call from a thread or subprocess)."""
    import serve.api as api_module
    import uvicorn

    api_module._weights_path = ROOT / weights
    uvicorn.run(api_module.app, host=host, port=port, log_level="warning")


def _start_vite(frontend_dir: Path) -> subprocess.Popen:
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    return subprocess.Popen(
        [npm, "run", "dev"],
        cwd=frontend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _mount_prod_frontend(frontend_dist: Path, api_port: int) -> None:
    """Mount the built React app as static files on the FastAPI app."""
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    import serve.api as api_module

    @api_module.app.get("/")
    async def _index():
        return FileResponse(frontend_dist / "index.html")

    # Mount assets first so /assets/* is handled before the catch-all
    api_module.app.mount(
        "/assets",
        StaticFiles(directory=frontend_dist / "assets"),
        name="frontend-assets",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="nathwaniGPT TUI")
    parser.add_argument("--weights",  default="weights/phi4-mini")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8080)
    parser.add_argument("--prod",     action="store_true",
                        help="Serve the pre-built frontend from FastAPI instead of Vite dev server")
    args = parser.parse_args()

    frontend_dir  = Path(__file__).parent / "frontend"
    frontend_dist = frontend_dir / "dist"
    vite_proc: subprocess.Popen | None = None

    if args.prod:
        if not frontend_dist.exists():
            print("ERROR: dist/ not found. Build first:")
            print(f"  cd {frontend_dir} && npm install && npm run build")
            sys.exit(1)
        _mount_prod_frontend(frontend_dist, args.api_port)
        url = f"http://{args.host}:{args.api_port}"
        print(f"Serving built frontend from {frontend_dist}")
    else:
        # Check node_modules exist
        if not (frontend_dir / "node_modules").exists():
            print("Installing frontend dependencies…")
            npm = "npm.cmd" if sys.platform == "win32" else "npm"
            subprocess.run([npm, "install"], cwd=frontend_dir, check=True)

        print("Starting Vite dev server…")
        vite_proc = _start_vite(frontend_dir)
        url = "http://localhost:5173"

    # Start API in a background thread so we can open the browser first
    api_thread = threading.Thread(
        target=_start_api,
        args=(args.weights, args.host, args.api_port),
        daemon=True,
    )
    api_thread.start()

    print(f"API server starting on http://{args.host}:{args.api_port} …")
    print(f"Loading nathwaniGPT weights from {ROOT / args.weights} …")
    time.sleep(1.5)
    webbrowser.open(url)
    print(f"Opened {url}")
    print("Press Ctrl+C to stop.\n")

    try:
        api_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        if vite_proc is not None:
            vite_proc.terminate()


if __name__ == "__main__":
    main()
