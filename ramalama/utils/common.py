from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Optional

from ramalama.utils.logger import logger

if TYPE_CHECKING:
    from ramalama.types import SUPPORTED_ENGINES
    from ramalama.config import Config

from ramalama.utils.process import run_cmd

# Re-export from focused modules so existing `from ramalama.utils.common import X` still works
from ramalama.utils.crypto import generate_sha256, generate_sha256_binary, verify_checksum  # noqa: F401
from ramalama.utils.naming import (  # noqa: F401
    SPLIT_MODEL_PATH_RE,
    genname,
    is_split_file_model,
    rm_until_substring,
    sanitize_filename,
)
from ramalama.utils.process import (  # noqa: F401
    available,
    engine_version,
    exec_cmd,
    perror,
    populate_volume_from_image,
    quoted,
    run_cmd,
)
from ramalama.utils.gpu import (  # noqa: F401
    AccelImageArgs,
    AccelImageArgsOtherRuntime,
    AccelImageArgsWithImage,
    AccelType,
    GPUEnvVar,
    accel_image,
    attempt_to_use_versioned,
    get_accel,
    get_accel_env_vars,
    get_gpu_type_env_vars,
    minor_release,
    set_accel_env_vars,
    set_gpu_type_env_vars,
    tagged_image,
)

MNT_DIR = "/mnt/models"
MNT_FILE = f"{MNT_DIR}/model.file"
MNT_MMPROJ_FILE = f"{MNT_DIR}/mmproj.file"
MNT_FILE_DRAFT = f"{MNT_DIR}/draft_model.file"
MNT_CHAT_TEMPLATE_FILE = f"{MNT_DIR}/chat_template.file"

MIN_VRAM_BYTES = 1073741824  # 1GiB

podman_machine_accel = False


def _platform_needs_podman_machine() -> bool:
    return sys.platform in ("win32", "darwin")


def _list_podman_machines(engine: str) -> list[dict]:
    """List podman machines as parsed JSON. Returns empty list on failure."""
    cmd = [engine, "machine", "list", "--format", "json"]
    if sys.platform == "darwin":
        cmd.append("--all-providers")
    try:
        output = run_cmd(cmd, ignore_stderr=True, encoding="utf-8").stdout.strip()
        machines = json.loads(output)
        return machines if isinstance(machines, list) else []
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        logger.debug(f"Failed to list podman machines: {e}")
        return []


def _find_default_machine(machines: list[dict]) -> dict | None:
    for m in machines:
        if m.get("Default", False):
            return m
    return machines[0] if machines else None


def _start_podman_machine(engine: str, machine_name: str) -> bool:
    """Attempt to start a podman machine. Returns True on success."""
    perror(f"Podman machine '{machine_name}' is not running. Starting it now...")
    try:
        run_cmd([engine, "machine", "start", machine_name], ignore_stderr=False, stdout=None)
        perror(f"Podman machine '{machine_name}' started successfully.")
        return True
    except subprocess.CalledProcessError as e:
        perror(f"Failed to start podman machine '{machine_name}': {e}")
        return False


def ensure_podman_machine(engine: SUPPORTED_ENGINES, config: Config | None = None) -> bool:
    """Ensure a podman machine is running on platforms that require one (Windows, macOS).

    Returns True if podman is usable (machine running or not needed).
    Returns False if podman cannot be used (no machine, start failed, user declined).
    """
    if not _platform_needs_podman_machine():
        return True

    machines = _list_podman_machines(engine)

    if not machines:
        perror(
            "Podman is installed but no machine exists.\n"
            "Please create one first with:\n"
            f"  {engine} machine init\n"
            f"  {engine} machine start"
        )
        return False

    default_machine = _find_default_machine(machines)
    if default_machine is None:
        return False

    machine_name = default_machine.get("Name", "")
    is_running = default_machine.get("Running", False)

    if not is_running:
        if not _start_podman_machine(engine, machine_name):
            return False

        # Re-fetch machine info after start
        machines = _list_podman_machines(engine)
        default_machine = _find_default_machine(machines)
        if default_machine is None or not default_machine.get("Running", False):
            perror(f"Podman machine '{machine_name}' failed to reach running state.")
            return False

    # macOS-specific GPU provider check
    if sys.platform == "darwin":
        result = _handle_macos_provider(default_machine, config)
        if result is not None:
            return result

    return True


def _handle_macos_provider(machine: dict, config: Config | None = None) -> bool | None:
    """Handle macOS-specific GPU provider checks. Returns None to defer to caller."""
    global podman_machine_accel
    provider = machine.get("VMType", "")

    if provider == "applehv":
        if config is not None and config.user.no_missing_gpu_prompt:
            return True
        return _confirm_no_gpu(machine.get("Name", ""), provider)

    if "krun" in provider:
        podman_machine_accel = True
        return True

    return None


def _confirm_no_gpu(name, provider) -> bool:
    while True:
        user_input = (
            input(
                f"Warning! Your VM {name} is using {provider}, which does not support GPU. "
                "Only the provider libkrun has GPU support. "
                "See `man ramalama-macos` for more information. "
                "Do you want to proceed without GPU? (yes/no): "
            )
            .strip()
            .lower()
        )
        if user_input in ["yes", "y"]:
            return True
        if user_input in ["no", "n"]:
            return False
        print("Invalid input. Please enter 'yes' or 'no'.")


# Keep old name as alias for backward compatibility
def confirm_no_gpu(name, provider) -> bool:
    return _confirm_no_gpu(name, provider)


def handle_provider(machine, config: Config | None = None) -> bool | None:
    return _handle_macos_provider(machine, config)


def apple_vm(engine: SUPPORTED_ENGINES, config: Config | None = None) -> bool:
    return ensure_podman_machine(engine, config)


class ContainerEntryPoint(str):
    def __init__(self, entrypoint: Optional[str] = None):
        self.entrypoint = entrypoint

    def __str__(self):
        return str(self.entrypoint)

    def __repr__(self):
        return repr(self.entrypoint)
