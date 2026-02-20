from ramalama.cli._utils import suppressCompleter


def register(subparsers):
    parser = subparsers.add_parser("containers", aliases=["ps"], help="list all RamaLama containers")
    parser.add_argument(
        "--format", help="pretty-print containers to JSON or using a Go template", completer=suppressCompleter
    )
    parser.add_argument("-n", "--noheading", dest="noheading", action="store_true", help="do not display heading")
    parser.add_argument("--no-trunc", dest="notrunc", action="store_true", help="display the extended information")
    parser.set_defaults(func=list_containers)


def list_containers(args):
    from ramalama.runtime import engine

    containers = engine.containers(args)
    if len(containers) == 0:
        return
    print("\n".join(containers))
