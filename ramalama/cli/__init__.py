import argparse
import errno
import subprocess
import sys
import urllib.error

from ramalama.cli._parser import ArgumentParserWithDefaults, get_initial_parser, get_parser
from ramalama.cli._utils import post_parse_setup
from ramalama.utils.common import perror
from ramalama.config import get_config
from ramalama.utils.endian import EndianMismatchError
from ramalama.model_inspect.error import ParseError
from ramalama.transports.base import (
    NoGGUFModelFileFound,
    NoRefFileFound,
)


def init_cli():
    """Initialize the RamaLama CLI and parse command line arguments."""
    return parse_args_from_cmd(sys.argv[1:])


def parse_args_from_cmd(cmd: list[str]) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parse arguments based on a command string"""
    config = get_config()
    if any(arg in ("--dryrun", "--dry-run", "--generate") or arg.startswith("--generate=") for arg in sys.argv[1:]):
        config.dryrun = True

    initial_parser = get_initial_parser()
    initial_args, _ = initial_parser.parse_known_args(cmd)
    for arg in initial_args.__dict__.keys():
        if hasattr(config, arg):
            setattr(config, arg, getattr(initial_args, arg))

    parser = get_parser()
    args = parser.parse_args(cmd)
    post_parse_setup(args)
    return parser, args


def main() -> None:
    def eprint(e: Exception | str, exit_code: int):
        try:
            if args.debug:
                from ramalama.utils.logger import logger

                logger.exception(e)
        except Exception:
            pass
        perror("Error: " + str(e).strip("'\""))
        sys.exit(exit_code)

    parser: ArgumentParserWithDefaults
    args: argparse.Namespace
    try:
        parser, args = init_cli()
        try:
            import argcomplete

            argcomplete.autocomplete(parser)
        except Exception:
            pass

        if not args.subcommand:
            parser.print_usage()
            perror("ramalama: requires a subcommand")
            return

        args.func(args)
    except urllib.error.HTTPError as e:
        eprint(f"pulling {e.geturl()} failed: {e}", errno.EINVAL)
    except FileNotFoundError as e:
        eprint(e, errno.ENOENT)
    except (ConnectionError, IndexError, KeyError, ValueError, NoRefFileFound) as e:
        eprint(e, errno.EINVAL)
    except NotImplementedError as e:
        eprint(e, errno.ENOSYS)
    except subprocess.TimeoutExpired as e:
        eprint(e, errno.ETIMEDOUT)
    except subprocess.CalledProcessError as e:
        eprint(e, e.returncode)
    except EndianMismatchError:
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except IOError as e:
        eprint(e, errno.EIO)
    except ParseError as e:
        eprint(f"Failed to parse model: {e}", errno.EINVAL)
    except NoGGUFModelFileFound:
        eprint(f"No GGUF model file found for downloaded model '{args.model}'", errno.ENOENT)  # type: ignore
    except Exception as e:
        if isinstance(e, OSError) and hasattr(e, "winerror") and e.winerror == 206:
            eprint("Path too long, please enable long path support in the Windows registry", errno.ENAMETOOLONG)
        else:
            raise
