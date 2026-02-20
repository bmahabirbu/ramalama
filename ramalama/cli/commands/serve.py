import argparse

from ramalama.cli._parser import CoerceToBool, OverrideDefaultAction, default_image
from ramalama.cli._utils import (
    GENERATE_OPTIONS,
    assemble_command_lazy,
    default_threads,
    local_env,
    local_images,
    local_models,
    parse_generate_option,
    parse_port_option,
    suppressCompleter,
)
from ramalama.config import get_config
from ramalama.transports.base import compute_serving_port
from ramalama.transports.transport_factory import New, TransportFactory


def register(subparsers):
    parser = subparsers.add_parser("serve", help="serve REST API on specified AI Model")
    runtime_options(parser)
    parser.add_argument("MODEL", completer=local_models)
    parser.set_defaults(func=serve_cli)


def serve_cli(args):
    if not args.container:
        args.detach = False

    try:
        args.port = compute_serving_port(args)
        model = New(args.MODEL, args)
        model.ensure_model_exists(args)
    except KeyError as e:
        try:
            if "://" in args.MODEL:
                raise e
            args.quiet = True
            model = TransportFactory(args.MODEL, args, ignore_stderr=True).create_oci()
            model.ensure_model_exists(args)
            args.MODEL = f"oci://{args.MODEL}"
        except Exception:
            raise e

    model.serve(args, assemble_command_lazy(args))


def runtime_options(parser):
    config = get_config()
    parser.add_argument("--authfile", help="path of the authentication file")
    parser.add_argument(
        "--cache-reuse",
        dest="cache_reuse",
        type=int,
        default=config.cache_reuse,
        help="min chunk size to attempt reusing from the cache via KV shifting",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "-c",
        "--ctx-size",
        dest="context",
        type=int,
        default=config.ctx_size,
        help="size of the prompt context (0 = loaded from model)",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "--max-model-len",
        dest="context",
        type=int,
        default=config.ctx_size,
        help=argparse.SUPPRESS,
        completer=suppressCompleter,
    )
    parser.add_argument(
        "-d", "--detach", action="store_true", dest="detach", help="run the container in detached mode"
    )
    parser.add_argument(
        "--device",
        dest="device",
        action='append',
        type=str,
        help="device to leak in to the running container (or 'none' to pass no device)",
    )
    parser.add_argument(
        "--env",
        dest="env",
        action='append',
        type=str,
        default=config.env,
        help="environment variables to add to the running container",
        completer=local_env,
    )
    parser.add_argument(
        "--generate",
        type=parse_generate_option,
        choices=GENERATE_OPTIONS,
        help="generate specified configuration format for running the AI Model as a service",
    )
    parser.add_argument(
        "--add-to-unit",
        dest="add_to_unit",
        action='append',
        type=str,
        help="add KEY VALUE pair to generated unit file in the section SECTION (only valid with --generate)",
        metavar="SECTION:KEY:VALUE",
    )
    parser.add_argument(
        "--host",
        default=config.host,
        help="IP address to listen",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "--image",
        default=default_image(),
        help="OCI container image to run with the specified AI model",
        action=OverrideDefaultAction,
        completer=local_images,
    )
    parser.add_argument(
        "--keep-groups",
        dest="podman_keep_groups",
        default=config.keep_groups,
        action="store_true",
        help="""pass `--group-add keep-groups` to podman.
If GPU device on host is accessible to via group access, this option leaks the user groups into the container.""",
    )
    parser.add_argument("--model-draft", help="Draft model", completer=local_models)
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        help="name of container in which the Model will be run",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=config.max_tokens,
        help="maximum number of tokens to generate (0 = unlimited)",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "--network",
        "--net",
        type=str,
        help="set the network mode for the container",
    )
    parser.add_argument(
        "--ngl",
        dest="ngl",
        type=int,
        default=config.ngl,
        help="number of layers to offload to the gpu, if available",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "--thinking",
        default=config.thinking,
        help="enable/disable thinking mode in reasoning models",
        action=CoerceToBool,
    )
    parser.add_argument(
        "--oci-runtime",
        help="override the default OCI runtime used to launch the container",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "-p",
        "--port",
        type=parse_port_option,
        default=config.port,
        action=OverrideDefaultAction,
        help="port for AI Model server to listen on",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "--privileged", dest="privileged", action="store_true", help="give extended privileges to container"
    )
    parser.add_argument(
        "--pull",
        dest="pull",
        type=str,
        default=config.pull,
        choices=["always", "missing", "never", "newer"],
        help='pull image policy',
    )
    parser.add_argument(
        "--runtime-args",
        dest="runtime_args",
        default="",
        type=str,
        help="arguments to add to runtime invocation",
        completer=suppressCompleter,
    )
    parser.add_argument("--seed", help="override random seed", completer=suppressCompleter)
    parser.add_argument(
        "--selinux",
        default=config.selinux,
        action=CoerceToBool,
        help="Enable SELinux container separation",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=config.temp,
        help="temperature of the response from the AI model",
        completer=suppressCompleter,
    )
    def_threads = default_threads()
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=def_threads,
        help=f"number of cpu threads to use, the default is {def_threads} on this system, -1 means use this default",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "--tls-verify",
        dest="tlsverify",
        default=True,
        help="require HTTPS and verify certificates when contacting registries",
    )
    parser.add_argument(
        "--webui",
        dest="webui",
        choices=["on", "off"],
        default="on",
        help="enable or disable the web UI (default: on)",
    )
    parser.add_argument(
        "--dri",
        dest="dri",
        choices=["on", "off"],
        default="on",
        help="mount /dev/dri into the container for GPU access (default: on)",
    )
