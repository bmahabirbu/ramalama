"""Docker Compose YAML generator."""

import os
import shlex
from typing import Optional, Tuple

from ramalama.utils.common import get_accel_env_vars
from ramalama.utils.file import PlainFile
from ramalama.version import version


class Compose:
    def __init__(
        self,
        model_name: str,
        model_paths: Tuple[str, str],
        chat_template_paths: Optional[Tuple[str, str]],
        mmproj_paths: Optional[Tuple[str, str]],
        args,
        exec_args,
    ):
        self.src_model_path, self.dest_model_path = model_paths
        self.src_chat_template_path, self.dest_chat_template_path = (
            chat_template_paths if chat_template_paths is not None else ("", "")
        )
        self.src_mmproj_path, self.dest_mmproj_path = mmproj_paths if mmproj_paths is not None else ("", "")
        self.src_model_path = self.src_model_path.removeprefix("oci://")

        self.model_name = model_name
        custom_name = getattr(args, "name", None)
        self.name = custom_name if custom_name else f"ramalama-{model_name}"
        self.args = args
        self.exec_args = exec_args
        self.image = args.image

    def _gen_volumes(self) -> str:
        volumes = "    volumes:"
        volumes += self._gen_model_volume()
        if self.src_chat_template_path and os.path.exists(self.src_chat_template_path):
            volumes += self._gen_chat_template_volume()
        if self.src_mmproj_path and os.path.exists(self.src_mmproj_path):
            volumes += self._gen_mmproj_volume()
        return volumes

    def _gen_model_volume(self) -> str:
        return f'\n      - "{self.src_model_path}:{self.dest_model_path}:ro"'

    def _gen_chat_template_volume(self) -> str:
        return f'\n      - "{self.src_chat_template_path}:{self.dest_chat_template_path}:ro"'

    def _gen_mmproj_volume(self) -> str:
        return f'\n      - "{self.src_mmproj_path}:{self.dest_mmproj_path}:ro"'

    def _gen_devices(self) -> str:
        device_list = []
        for dev_path in ["/dev/dri", "/dev/kfd", "/dev/accel"]:
            if os.path.exists(dev_path):
                device_list.append(dev_path)
        if not device_list:
            return ""
        devices_str = "    devices:"
        for dev in device_list:
            devices_str += f'\n      - "{dev}:{dev}"'
        return devices_str

    def _gen_ports(self) -> str:
        port_arg = getattr(self.args, "port", None)
        if not port_arg:
            return '    ports:\n      - "8080:8080"'
        p = port_arg.split(":", 2)
        host_port = p[1] if len(p) > 1 else p[0]
        container_port = p[0]
        return f'    ports:\n      - "{host_port}:{container_port}"'

    def _gen_environment(self) -> str:
        env_vars = get_accel_env_vars()
        if getattr(self.args, "env", None):
            for e in self.args.env:
                key, val = e.split("=", 1)
                env_vars[key] = val
        if not env_vars:
            return ""
        env_spec = "    environment:"
        for k, v in env_vars.items():
            env_spec += f'\n      - {k}={v}'
        return env_spec

    def _gen_gpu_deployment(self) -> str:
        gpu_keywords = ["cuda", "rocm", "gpu"]
        if not any(keyword in self.image.lower() for keyword in gpu_keywords):
            return ""
        return """\
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]"""

    def _gen_command(self) -> str:
        if not self.exec_args:
            return ""
        cmd = shlex.join(self.exec_args)
        return f"    command: {cmd}"

    def generate(self) -> PlainFile:
        _version = version()
        volumes_string = self._gen_volumes()
        ports_string = self._gen_ports()
        environment_string = self._gen_environment()
        devices_string = self._gen_devices()
        gpu_deploy_string = self._gen_gpu_deployment()
        command_string = self._gen_command()

        content = f"""\
# Save this output to a 'docker-compose.yaml' file and run 'docker compose up'.
#
# Created with ramalama-{_version}

services:
  {self.model_name}:
    container_name: {self.name}
    image: {self.image}
{volumes_string}
{ports_string}
{environment_string}
{devices_string}
{gpu_deploy_string}
{command_string}
    restart: unless-stopped
"""
        content = "\n".join(line for line in content.splitlines() if line.strip())
        return genfile(self.name, content)


def genfile(name: str, content: str) -> PlainFile:
    file_name = "docker-compose.yaml"
    print(f"Generating Compose YAML file: {file_name}")
    file = PlainFile(file_name)
    file.content = content
    return file
