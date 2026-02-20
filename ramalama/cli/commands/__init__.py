import importlib.metadata

from ramalama.cli.commands import containers, list, pull, rm, serve, stop
from ramalama.utils.logger import logger

_BUILTINS = [serve, pull, list, rm, stop, containers]


def register_builtins(subparsers):
    for mod in _BUILTINS:
        mod.register(subparsers)


def discover_plugins(subparsers):
    for ep in importlib.metadata.entry_points(group="ramalama.commands"):
        try:
            plugin = ep.load()
            if callable(plugin):
                plugin(subparsers)
            elif hasattr(plugin, "register"):
                plugin.register(subparsers)
        except Exception as e:
            logger.warning(f"Failed to load ramalama plugin '{ep.name}': {e}")
