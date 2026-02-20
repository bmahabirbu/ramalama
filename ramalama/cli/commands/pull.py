from ramalama.cli._parser import CoerceToBool
from ramalama.cli._utils import suppressCompleter
from ramalama.config import get_config
from ramalama.transports.transport_factory import New


def register(subparsers):
    config = get_config()
    parser = subparsers.add_parser("pull", help="pull AI Model from Model registry to local storage")
    parser.add_argument("--authfile", help="path of the authentication file")
    parser.add_argument(
        "--tls-verify",
        dest="tlsverify",
        default=True,
        help="require HTTPS and verify certificates when contacting registries",
    )
    parser.add_argument(
        "--verify",
        default=config.verify,
        action=CoerceToBool,
        help="verify the model after pull, disable to allow pulling of models with different endianness",
    )
    parser.add_argument("MODEL", completer=suppressCompleter)
    parser.set_defaults(func=pull_cli)


def pull_cli(args):
    model = New(args.MODEL, args)
    model.pull(args)
