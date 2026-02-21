"""GPU detection, acceleration env vars, and container image selection."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, get_args

from ramalama.utils.logger import logger
from ramalama.utils.process import perror, run_cmd
from ramalama.version import version

if TYPE_CHECKING:
    from ramalama.config import Config, RamalamaImageConfig


AccelType: TypeAlias = Literal["vulkan"]


@lru_cache(maxsize=1)
def get_accel() -> AccelType | Literal["none"]:
    if sys.platform == "win32":
        return "vulkan"
    if os.path.exists("/dev/dxg"):
        return "vulkan"
    if os.path.exists("/dev/dri"):
        return "vulkan"
    return "none"


def set_accel_env_vars():
    if get_accel_env_vars():
        return
    get_accel()


def set_gpu_type_env_vars():
    if get_gpu_type_env_vars():
        return
    get_accel()


GPUEnvVar: TypeAlias = Literal[
    "GGML_VK_VISIBLE_DEVICES",
]


def get_gpu_type_env_vars() -> dict[GPUEnvVar, str]:
    return {k: v for k in get_args(GPUEnvVar) if (v := os.environ.get(k))}


def get_accel_env_vars() -> dict[GPUEnvVar, str]:
    return get_gpu_type_env_vars()


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def minor_release() -> str:
    version_split = version().split(".")
    vers = ".".join(version_split[:2])
    if vers == "0":
        vers = "latest"
    return vers


def tagged_image(image: str) -> str:
    if len(image.split(":")) > 1:
        return image
    return f"{image}:{minor_release()}"


class AccelImageArgsWithImage(Protocol):
    image: str


class AccelImageArgsOtherRuntime(Protocol):
    runtime: str
    container: bool
    quiet: bool


AccelImageArgs: TypeAlias = None | AccelImageArgsOtherRuntime


def accel_image(config: Config, images: RamalamaImageConfig | None = None, conf_key: str = "image") -> str:
    """Selects the appropriate container image based on config, arguments, environment."""
    if config.is_set(conf_key):
        return tagged_image(getattr(config, conf_key))

    if not images:
        images = config.images

    set_gpu_type_env_vars()
    gpu_type = next(iter(get_gpu_type_env_vars()), "")

    image = images.get(gpu_type, getattr(config, f"default_{conf_key}"))

    if ":" in image:
        return image

    vers = minor_release()
    should_pull = config.pull in ["always", "missing"] and not config.dryrun
    if config.engine and attempt_to_use_versioned(config.engine, image, vers, True, should_pull):
        return f"{image}:{vers}"

    return f"{image}:latest"


def attempt_to_use_versioned(conman: str, image: str, vers: str, quiet: bool, should_pull: bool) -> bool:
    try:
        if run_cmd([conman, "inspect", f"{image}:{vers}"], ignore_all=True):
            return True
    except Exception:
        pass

    if not should_pull:
        return False

    try:
        if not quiet:
            perror(f"Attempting to pull {image}:{vers} ...")
        run_cmd([conman, "pull", f"{image}:{vers}"], ignore_stderr=True)
        return True
    except Exception:
        return False
