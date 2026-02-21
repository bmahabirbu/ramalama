import os
import shlex
from functools import lru_cache
from urllib.parse import urlparse

from ramalama.cli._arg_normalization import normalize_pull_arg
from ramalama.config import get_config
from ramalama.utils.log_levels import LogLevel
from ramalama.utils.logger import configure_logger, logger
from ramalama.utils.shortnames import Shortnames
from ramalama.transports.transport_factory import New

# if autocomplete doesn't exist, just do nothing, don't break
try:
    import argcomplete

    suppressCompleter: type[argcomplete.completers.SuppressCompleter] | None = argcomplete.completers.SuppressCompleter
except Exception:
    suppressCompleter = None


GENERATE_OPTIONS = ["quadlet", "kube", "quadlet/kube", "compose"]
LIST_SORT_FIELD_OPTIONS = ["size", "modified", "name"]
LIST_SORT_ORDER_OPTIONS = ["desc", "asc"]


@lru_cache(maxsize=1)
def get_shortnames():
    return Shortnames()


class ParsedGenerateInput:
    def __init__(self, gen_type: str, output_dir: str):
        self.gen_type = gen_type
        self.output_dir = output_dir

    def __str__(self):
        return self.gen_type

    def __repr__(self):
        return self.__str__()

    def __eq__(self, value):
        return self.gen_type == value


def parse_generate_option(option: str) -> ParsedGenerateInput:
    generate, output_dir = option, "."
    if generate.count(":") > 0:
        generate, output_dir = generate.split(":", 1)
    if output_dir == "":
        output_dir = "."

    return ParsedGenerateInput(generate, output_dir)


def parse_port_option(option: str) -> str:
    port = int(option)
    if port <= 0 or port >= 65535:
        raise ValueError(f"Invalid port '{port}'")
    return option


def default_threads():
    config = get_config()
    if config.threads < 0:
        nproc = os.cpu_count()
        if nproc and nproc > 4:
            return int(nproc / 2)
        return 4
    return config.threads


def human_duration(d):
    if d < 1:
        return "Less than a second"
    if d == 1:
        return "1 second"
    if d < 60:
        return f"{d} seconds"
    if d < 120:
        return "1 minute"
    if d < 3600:
        return f"{d // 60} minutes"
    if d < 7200:
        return "1 hour"
    if d < 86400:
        return f"{d // 3600} hours"
    if d < 172800:
        return "1 day"
    if d < 604800:
        return f"{d // 86400} days"
    if d < 1209600:
        return "1 week"
    if d < 2419200:
        return f"{d // 604800} weeks"
    if d < 4838400:
        return "1 month"
    if d < 31536000:
        return f"{d // 2419200} months"
    return "1 year" if d < 63072000 else f"{d // 31536000} years"


def human_readable_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            size = round(size, 2)
            return f"{size} {unit}"
        size /= 1024
    return f"{size} PB"


def local_env(**kwargs):
    return os.environ


def local_models(prefix, parsed_args, **kwargs):
    from ramalama.cli.commands.list import _list_models

    return [model['name'] for model in _list_models(parsed_args)]


def local_containers(prefix, parsed_args, **kwargs):
    from ramalama.runtime import engine

    parsed_args.format = '{{.Names}}'
    return engine.containers(parsed_args)


def local_images(prefix, parsed_args, **kwargs):
    parsed_args.format = "{{.Repository}}:{{.Tag}}"
    from ramalama.runtime import engine

    return engine.images(parsed_args)


def post_parse_setup(args):
    """Perform additional setup after parsing arguments."""

    def map_https_to_transport(input: str) -> str:
        if input.startswith("https://") or input.startswith("http://"):
            url = urlparse(input)
            if url.path.count("/") != 2:
                return input
            if url.hostname in ["hf.co", "huggingface.co"]:
                return f"hf:/{url.path}"
            if url.hostname in ["ollama.com"]:
                return f"ollama:/{url.path}"
        return input

    if getattr(args, "MODEL", None):
        shortnames = get_shortnames()
        if isinstance(args.MODEL, str):
            args.INITIAL_MODEL = args.MODEL
            args.MODEL = map_https_to_transport(args.MODEL)
            args.UNRESOLVED_MODEL = args.MODEL
            args.MODEL = shortnames.resolve(args.MODEL)

        if isinstance(args.MODEL, list):
            args.INITIAL_MODEL = [m for m in args.MODEL]
            for i in range(len(args.MODEL)):
                args.MODEL[i] = map_https_to_transport(args.MODEL[i])
            args.UNRESOLVED_MODEL = [m for m in args.MODEL]
            for i in range(len(args.MODEL)):
                args.MODEL[i] = shortnames.resolve(args.MODEL[i])

        if not hasattr(args, "model"):
            args.model = args.MODEL

    if hasattr(args, "add_to_unit") and (add_to_units := args.add_to_unit):
        if getattr(args, "generate", None) is None:
            from ramalama.cli._parser import get_parser

            parser = get_parser()
            parser.error("--add-to-unit can only be used with --generate")
        if not (all(len([value for value in unit_to_add.split(":", 2) if value]) == 3 for unit_to_add in add_to_units)):
            from ramalama.cli._parser import get_parser

            parser = get_parser()
            parser.error("--add-to-unit parameters must be of the form <section>:<key>:<value>")

    if hasattr(args, "runtime_args"):
        args.runtime_args = shlex.split(args.runtime_args)

    if getattr(args, "runtime", None) == "mlx":
        if getattr(args, "container", None) is True:
            logger.info("MLX runtime automatically uses --nocontainer mode")
        args.container = False

    if hasattr(args, 'pull'):
        args.pull = normalize_pull_arg(args.pull, getattr(args, 'engine', None))

    if args.debug:
        log_level = LogLevel.DEBUG
    elif getattr(args, 'quiet', False):
        log_level = LogLevel.ERROR
    else:
        log_level = get_config().log_level or LogLevel.WARNING
    configure_logger(log_level)


def assemble_command_lazy(cli_args):
    from ramalama.inference.factory import assemble_command

    return assemble_command(cli_args)
