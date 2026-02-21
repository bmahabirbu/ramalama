# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RamaLama is a CLI tool for managing and serving AI models using containers. It provides a container-centric approach to AI model management, supporting multiple model registries (Hugging Face, Ollama, OCI registries) and automatic GPU detection with appropriate container image selection.

## Build and Development Commands

### Setup
```bash
make install-requirements    # Install dev dependencies via pip
```

### Testing
```bash
# Unit tests (pytest via tox)
make unit-tests              # Run unit tests
make unit-tests-verbose      # Run with full trace output
tox                          # Direct tox invocation

# E2E tests (pytest)
make e2e-tests               # Run with Podman (default)
make e2e-tests-docker        # Run with Docker
make e2e-tests-nocontainer   # Run without container engine

# System tests (BATS)
make bats                    # Run BATS system tests
make bats-nocontainer        # Run in nocontainer mode
make bats-docker             # Run with Docker

# All tests
make tests                   # Run unit tests and system-level integration tests
```

### Running a single test
```bash
# Unit test
tox -- test/unit/test_cli.py::test_function_name -vvv

# E2E test
tox -e e2e -- test/e2e/test_basic.py::test_function_name -vvv

# Single BATS file
RAMALAMA=$(pwd)/bin/ramalama bats -T test/system/030-run.bats
```

### Code Quality
```bash
make validate                # Run all validation (codespell, lint, format check, man-check, type check)
make lint                    # Run ruff + shellcheck
make check-format            # Check ruff formatting + import sorting
make format                  # Auto-format with ruff + import sorting
make type-check              # Run mypy type checking
make codespell               # Check spelling
```

### Documentation
```bash
make docs                    # Build manpages and docsite
```

## Architecture

### Source Structure (`ramalama/`)
- `__init__.py` - Package entry point
- `version.py` - Version string
- `types.py` - Shared type definitions (protocols, dataclasses for CLI/engine/store args)

### CLI (`ramalama/cli/`)
CLI entry point, argument parsing, and subcommand dispatch:
- `__init__.py` - Main entry point (`init_cli()`, `main()`)
- `_parser.py` - Argument parser setup, global flags, subcommand registration
- `_utils.py` - Shared CLI helpers, completers, `post_parse_setup()`
- `_arg_normalization.py` - Argument value normalization
- `commands/` - Subcommand implementations (`serve`, `pull`, `list`, `rm`, `stop`, `containers`)

### Configuration (`ramalama/config/`)
- `_loader.py` - Config loading, defaults, platform-specific paths, `get_config()`
- `types.py` - Config type aliases (`SUPPORTED_ENGINES`, `SUPPORTED_RUNTIMES`, `PathStr`)
- `layered.py` - Layered config merge logic

### Inference Engine (`ramalama/inference/`)
Builds runtime commands from YAML inference specs:
- `factory.py` - `assemble_command()` builds runtime commands (llama.cpp, mlx)
- `context.py` - Command execution context (`RamalamaCommandContext`)
- `schema.py` - Inference spec schema (`CommandSpecV1`)
- `error.py` - `InvalidInferenceEngineSpecError`

### Transport System (`ramalama/transports/`)
Handles pulling/pushing models from different registries:
- `base.py` - Base `Transport` class defining the interface
- `transport_factory.py` - `New()` and `TransportFactory` for creating transports
- `huggingface.py`, `ollama.py`, `oci.py`, `modelscope.py`, `rlcr.py` - Registry-specific implementations
- `hf_repo_base.py` - Shared base for HuggingFace-style registries (HF and ModelScope)
- `url.py` - HTTP/HTTPS/file URL transport
- `annotations.py` - OCI annotation constants
- Transports are selected via URL scheme prefixes: `huggingface://`, `ollama://`, `oci://`, etc.

### Model Store (`ramalama/model_store/`)
Manages local model storage:
- `global_store.py` - `GlobalModelStore` for listing models across the store
- `store.py` - Per-model store operations (pull, remove, snapshots, checksums)
- `reffile.py` - Reference file formats and migration
- `snapshot_file.py` - Snapshot file representation and validation
- `constants.py` - Store layout constants (`blobs/`, `refs/`, `snapshots/`)

### Model Inspect (`ramalama/model_inspect/`)
GGUF binary parsing and model info:
- `gguf_parser.py` - Low-level GGUF format parsing
- `gguf_info.py` - GGUF model info types, chat template extraction
- `base_info.py` - Base model info types
- `error.py` - `ParseError`

### Templates (`ramalama/templates/`)
Chat template format conversion:
- `conversion.py` - Go-to-Jinja conversion, OpenAI compatibility wrapping
- `go2jinja.py` - Go template parser and Jinja converter

### Runtime (`ramalama/runtime/`)
Container engine abstraction and deployment config generators:
- `engine.py` - Container engine abstraction (Podman/Docker): `run`, `build`, `stop`, health checks
- `generators/` - Deployment config generators:
  - `kube.py` - Kubernetes Deployment YAML
  - `quadlet.py` - Podman Quadlet unit files
  - `compose.py` - Docker Compose YAML

### Utilities (`ramalama/utils/`)
Shared utility functions split into focused modules:
- `gpu.py` - GPU detection (`get_accel()`), acceleration env vars, container image selection (`accel_image()`)
- `process.py` - Subprocess helpers (`run_cmd()`, `exec_cmd()`, `perror()`)
- `crypto.py` - SHA-256 checksums (`generate_sha256()`, `verify_checksum()`)
- `naming.py` - Name generation (`genname()`, `sanitize_filename()`, `rm_until_substring()`)
- `common.py` - Remaining helpers (mount paths, `ContainerEntryPoint`, `apple_vm()`) + re-exports from above
- `http_client.py` - HTTP downloads with resume, progress, retries
- `path_utils.py` - Cross-platform path handling
- `logger.py` - Logging setup
- `console.py` - Terminal/console helpers
- `proxy.py` - Proxy configuration
- `shortnames.py` - Model shortname resolution
- `endian.py` - GGUF endianness detection
- `compat.py` - Python version compatibility shims
- `file.py` - File abstractions (locking, INI-style unit files)
- `toml_parser.py` - Simple TOML parser for config files
- `log_levels.py` - Log level enum and coercion

### Key Patterns
- **GPU Detection**: `get_accel()` in `utils/gpu.py` detects GPU type and selects appropriate container image
- **Container Images**: GPU-specific images at `quay.io/ramalama/{ramalama,cuda,rocm,intel-gpu,...}`
- **Inference Engines**: llama.cpp (default), mlx (macOS only) - configured via YAML specs in `inference-spec/engines/`

## Test Structure

- `test/unit/` - pytest unit tests (fast, no external dependencies)
- `test/e2e/` - pytest end-to-end tests (marked with `@pytest.mark.e2e`)
- `test/system/` - BATS shell tests for full CLI integration testing

## Code Style

- Python 3.10+ required
- Line length: 120 characters
- Formatting: ruff format + ruff check (I rules)
- Type hints encouraged (mypy checked)
- Commits require DCO sign-off (`git commit -s`)
