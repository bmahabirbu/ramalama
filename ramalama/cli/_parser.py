import argparse
import os
from functools import lru_cache
from typing import get_args

from ramalama.utils.common import accel_image, get_accel
from ramalama.config import (
    SUPPORTED_ENGINES,
    SUPPORTED_RUNTIMES,
    coerce_to_bool,
    get_config,
)


@lru_cache(maxsize=1)
def default_image() -> str:
    return accel_image(get_config())


class ArgumentParserWithDefaults(argparse.ArgumentParser):
    def add_argument(self, *args, help=None, default=None, completer=None, **kwargs):
        if help is not None:
            kwargs['help'] = help
        if default is not None and args[0] != '-h':
            kwargs['default'] = default
            if help is not None and help != "==SUPPRESS==":
                kwargs['help'] += f' (default: {default})'
        action = super().add_argument(*args, **kwargs)
        if completer is not None:
            action.completer = completer  # type: ignore[attr-defined]
        return action


class OverrideDefaultAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + '_override', True)


class CoerceToBool(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, coerce_to_bool(values))


def get_description():
    """Return the description of the RamaLama tool."""
    return """\
RamaLama tool facilitates local management and serving of AI Models.

On first run RamaLama inspects your system for GPU support, falling back to CPU support if no GPUs are present.

RamaLama uses container engines like Podman or Docker to pull the appropriate OCI image with all of the software \
necessary to run an AI Model for your systems setup.

Running in containers eliminates the need for users to configure the host system for AI. After the initialization, \
RamaLama runs the AI Models within a container based on the OCI image.

RamaLama then pulls AI Models from model registries. Starting a chatbot or a rest API service from a simple single \
command. Models are treated similarly to how Podman and Docker treat container images.

When both Podman and Docker are installed, RamaLama defaults to Podman. The `RAMALAMA_CONTAINER_ENGINE=docker` \
environment variable can override this behaviour. When neither are installed, RamaLama will attempt to run the model \
with software on the local system.
"""


def abspath(astring) -> str:
    return os.path.abspath(astring)


def configure_arguments(parser):
    """Configure the global command-line arguments for the parser."""
    config = get_config()
    verbosity_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--container",
        dest="container",
        default=config.container,
        action="store_true",
        help=argparse.SUPPRESS,
    )
    verbosity_group.add_argument(
        "--debug",
        action="store_true",
        help="display debug messages",
    )
    parser.add_argument(
        "--dryrun",
        "--dry-run",
        dest="dryrun",
        action="store_true",
        help="show container runtime command without executing it",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=config.engine,
        choices=get_args(SUPPORTED_ENGINES),
        help="""run RamaLama using the specified container engine.
The RAMALAMA_CONTAINER_ENGINE environment variable modifies default behaviour.""",
    )
    parser.add_argument(
        "--nocontainer",
        dest="container",
        default=not config.container,
        action="store_false",
        help="""do not run RamaLama in the default container.
The RAMALAMA_IN_CONTAINER environment variable modifies default behaviour.""",
    )
    verbosity_group.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        action="store_true",
        help="Reduce output verbosity (silences warnings, simplifies list output)",
    )
    parser.add_argument(
        "--runtime",
        default=config.runtime,
        choices=get_args(SUPPORTED_RUNTIMES),
        help="specify the runtime to use; valid options are 'llama.cpp', 'vllm', and 'mlx'",
    )
    parser.add_argument(
        "--store",
        default=config.store,
        type=abspath,
        help="store AI Models in the specified directory",
    )
    parser.add_argument(
        "--noout",
        help=argparse.SUPPRESS,
    )


def create_argument_parser(description: str, add_help: bool = True):
    """Create and configure the argument parser for the CLI."""
    parser = ArgumentParserWithDefaults(
        prog="ramalama",
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=add_help,
    )
    configure_arguments(parser)
    return parser


def configure_subcommands(parser):
    """Add subcommand parsers to the main argument parser."""
    from ramalama.cli.commands import register_builtins, discover_plugins

    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = False
    register_builtins(subparsers)
    discover_plugins(subparsers)


def get_initial_parser():
    description = get_description()
    parser = create_argument_parser(description, add_help=False)
    return parser


def get_parser():
    description = get_description()
    parser = create_argument_parser(description, add_help=True)
    configure_subcommands(parser)
    return parser
