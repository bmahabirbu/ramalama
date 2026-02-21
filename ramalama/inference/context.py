"""Command execution context for inference specs."""

import argparse
import os
from typing import Optional

from ramalama.transports.transport_factory import CLASS_MODEL_TYPES, New
from ramalama.utils.common import get_accel
from ramalama.utils.console import should_colorize


class RamalamaArgsContext:

    def __init__(self) -> None:
        self.cache_reuse: Optional[int] = None
        self.container: Optional[bool] = None
        self.ctx_size: Optional[int] = None
        self.debug: Optional[bool] = None
        self.host: Optional[str] = None
        self.gguf: Optional[str] = None
        self.logfile: Optional[str] = None
        self.max_tokens: Optional[int] = None
        self.model_draft: Optional[str] = None
        self.ngl: Optional[int] = None
        self.port: Optional[int] = None
        self.runtime_args: Optional[str] = None
        self.seed: Optional[int] = None
        self.temp: Optional[float] = None
        self.thinking: Optional[bool] = None
        self.threads: Optional[int] = None
        self.webui: Optional[bool] = None
        self.router_mode: bool = False
        self.models_max: int = 4

    @staticmethod
    def from_argparse(args: argparse.Namespace) -> "RamalamaArgsContext":
        ctx = RamalamaArgsContext()
        ctx.cache_reuse = getattr(args, "cache_reuse", None)
        ctx.container = getattr(args, "container", None)
        ctx.ctx_size = getattr(args, "context", None)
        ctx.debug = getattr(args, "debug", None)
        ctx.host = getattr(args, "host", None)
        ctx.gguf = getattr(args, "gguf", None)
        ctx.logfile = getattr(args, "logfile", None)
        ctx.max_tokens = getattr(args, "max_tokens", None)
        ctx.model_draft = getattr(args, "model_draft", None)
        ctx.ngl = getattr(args, "ngl", None)
        ctx.port = getattr(args, "port", None)
        ctx.runtime_args = getattr(args, "runtime_args", None)
        ctx.seed = getattr(args, "seed", None)
        ctx.temp = getattr(args, "temp", None)
        ctx.thinking = getattr(args, "thinking", None)
        ctx.threads = getattr(args, "threads", None)
        ctx.webui = getattr(args, "webui", None)
        ctx.router_mode = getattr(args, "router_mode", False)
        ctx.models_max = getattr(args, "models_max", 4)
        return ctx


class RamalamaModelContext:

    def __init__(self, model: CLASS_MODEL_TYPES, is_container: bool, should_generate: bool, dry_run: bool):
        self.model = model
        self.is_container = is_container
        self.should_generate = should_generate
        self.dry_run = dry_run

    @property
    def name(self) -> str:
        return f"{self.model.model_name}:{self.model.model_tag}"

    @property
    def alias(self) -> str:
        return self.model.model_alias

    @property
    def model_path(self) -> str:
        return self.model._get_entry_model_path(self.is_container, self.should_generate, self.dry_run)

    @property
    def mmproj_path(self) -> Optional[str]:
        return self.model._get_mmproj_path(self.is_container, self.should_generate, self.dry_run)

    @property
    def chat_template_path(self) -> Optional[str]:
        return self.model._get_chat_template_path(self.is_container, self.should_generate, self.dry_run)

    @property
    def draft_model_path(self) -> str:
        if getattr(self.model, "draft_model", None):
            assert self.model.draft_model
            return self.model.draft_model._get_entry_model_path(self.is_container, self.should_generate, self.dry_run)
        return ""


class RamalamaHostContext:

    def __init__(
        self, is_container: bool, uses_vulkan: bool, should_colorize: bool, rpc_nodes: Optional[str]
    ):
        self.is_container = is_container
        self.uses_vulkan = uses_vulkan
        self.should_colorize = should_colorize
        self.rpc_nodes = rpc_nodes


class RamalamaCommandContext:

    def __init__(
        self,
        args: RamalamaArgsContext,
        model: RamalamaModelContext | None,
        host: RamalamaHostContext,
    ):
        self.args = args
        self.model = model
        self.host = host

    @staticmethod
    def from_argparse(cli_args: argparse.Namespace) -> "RamalamaCommandContext":
        args = RamalamaArgsContext.from_argparse(cli_args)
        should_generate = getattr(cli_args, "generate", None) is not None
        dry_run = getattr(cli_args, "dryrun", False)
        is_container = getattr(cli_args, "container", True)
        if hasattr(cli_args, "MODEL") and cli_args.MODEL is not None:
            model = RamalamaModelContext(New(cli_args.MODEL, cli_args), is_container, should_generate, dry_run)
        elif hasattr(cli_args, "model"):
            model = cli_args.model
        else:
            model = None

        host = RamalamaHostContext(
            is_container,
            get_accel() == "vulkan",
            should_colorize(),
            os.getenv("RAMALAMA_LLAMACPP_RPC_NODES", None),
        )
        return RamalamaCommandContext(args, model, host)
