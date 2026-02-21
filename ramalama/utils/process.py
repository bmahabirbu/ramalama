"""Subprocess helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from typing import IO, TYPE_CHECKING, Any

from ramalama.utils.logger import logger

if TYPE_CHECKING:
    from argparse import Namespace

    from ramalama.transports.base import Transport


def perror(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def available(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def quoted(arr) -> str:
    """Return string with quotes around elements containing spaces."""
    return " ".join(['"' + element + '"' if ' ' in element else element for element in arr])


def exec_cmd(args, stdout2null: bool = False, stderr2null: bool = False):
    logger.debug(f"exec_cmd: {quoted(args)}")

    stdout_target = subprocess.DEVNULL if stdout2null else None
    stderr_target = subprocess.DEVNULL if stderr2null else None
    try:
        result = subprocess.run(args, stdout=stdout_target, stderr=stderr_target, check=False)
        sys.exit(result.returncode)
    except Exception as e:
        perror(f"Failed to execute {quoted(args)}: {e}")
        raise


def run_cmd(
    args: Sequence[str],
    cwd: str | None = None,
    stdout: int | IO[Any] | None = subprocess.PIPE,
    ignore_stderr: bool = False,
    ignore_all: bool = False,
    encoding: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Run the given command arguments."""
    logger.debug(f"run_cmd: {quoted(args)}")
    logger.debug(f"Working directory: {cwd}")
    logger.debug(f"Ignore stderr: {ignore_stderr}")
    logger.debug(f"Ignore all: {ignore_all}")
    logger.debug(f"env: {env}")

    serr = None
    if ignore_all or ignore_stderr:
        serr = subprocess.DEVNULL

    sout = stdout
    if ignore_all:
        sout = subprocess.DEVNULL

    if env:
        env = os.environ | env

    result = subprocess.run(args, check=True, cwd=cwd, stdout=sout, stderr=serr, encoding=encoding, env=env)
    logger.debug(f"Command finished with return code: {result.returncode}")

    return result


def engine_version(engine: str) -> str:
    cmd_args = [str(engine), "version", "--format", "{{ .Client.Version }}"]
    return run_cmd(cmd_args, encoding="utf-8").stdout.strip()


def populate_volume_from_image(model: Transport, args: Namespace, output_filename: str, src_model_dir: str = "models"):
    """Builds a Docker-compatible mount string that mirrors Podman image mounts for model assets."""
    vol_hash = hashlib.sha256(model.model.encode()).hexdigest()[:12]
    volume = f"ramalama-models-{vol_hash}"
    src = f"src-{vol_hash}"

    run_cmd([args.engine, "volume", "create", volume], ignore_stderr=True)
    run_cmd([args.engine, "rm", "-f", src], ignore_stderr=True)
    run_cmd([args.engine, "create", "--name", src, model.model])

    try:
        export_cmd = [args.engine, "export", src]
        untar_cmd = [
            args.engine,
            "run",
            "--rm",
            "-i",
            "--mount",
            f"type=volume,src={volume},dst=/mnt",
            "busybox",
            "tar",
            "-C",
            "/mnt",
            "--strip-components=1",
            "-x",
            "-p",
            "-f",
            "-",
            f"{src_model_dir}/{output_filename}",
        ]

        with (
            subprocess.Popen(export_cmd, stdout=subprocess.PIPE) as p_out,
            subprocess.Popen(untar_cmd, stdin=p_out.stdout) as p_in,
        ):
            p_out.stdout.close()  # type: ignore
            rc_in = p_in.wait()
            rc_out = p_out.wait()
            if rc_in != 0 or rc_out != 0:
                raise subprocess.CalledProcessError(rc_in or rc_out, untar_cmd if rc_in else export_cmd)
    finally:
        run_cmd([args.engine, "rm", "-f", src], ignore_stderr=True)

    return volume
