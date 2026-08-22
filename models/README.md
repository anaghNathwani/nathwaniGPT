# Models

Each subdirectory is a versioned snapshot of nathwaniGPT. To add a new version, create a new folder (`v2/`, `v3/`, etc.) with its own `Modelfile` and `README.md` describing what changed.

## Versions

| Version | Base Model | Date |
|---|---|---|
| [v1](v1/) | qwen2.5:14b | 2026-08-22 |

## Convention

- Folder name matches the version tag: `v1`, `v2`, etc.
- Each folder contains a `Modelfile` and a `README.md` changelog entry.
- The root `Modelfile` always points to the latest stable version.
