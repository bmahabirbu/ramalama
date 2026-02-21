from __future__ import annotations

import json
import subprocess
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


def confirm_no_gpu(name, provider) -> bool:
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


def handle_provider(machine, config: Config | None = None) -> bool | None:
    global podman_machine_accel
    name = machine.get("Name")
    provider = machine.get("VMType")
    running = machine.get("Running")
    if running:
        if provider == "applehv":
            if config is not None and config.user.no_missing_gpu_prompt:
                return True
            else:
                return confirm_no_gpu(name, provider)
        if "krun" in provider:
            podman_machine_accel = True
            return True

    return None


def apple_vm(engine: SUPPORTED_ENGINES, config: Config | None = None) -> bool:
    podman_machine_list = [engine, "machine", "list", "--format", "json", "--all-providers"]
    try:
        machines_json = run_cmd(podman_machine_list, ignore_stderr=True, encoding="utf-8").stdout.strip()
        machines = json.loads(machines_json)
        for machine in machines:
            result = handle_provider(machine, config)
            if result is not None:
                return result
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to list and parse podman machines: {e}")
    return False


class ContainerEntryPoint(str):
    def __init__(self, entrypoint: Optional[str] = None):
        self.entrypoint = entrypoint

    def __str__(self):
        return str(self.entrypoint)

    def __repr__(self):
        return repr(self.entrypoint)
