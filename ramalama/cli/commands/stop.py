from ramalama.cli._utils import local_containers, suppressCompleter


def register(subparsers):
    parser = subparsers.add_parser("stop", help="stop named container that is running AI Model")
    parser.add_argument("-a", "--all", action="store_true", help="stop all RamaLama containers")
    parser.add_argument(
        "--ignore", action="store_true", help="ignore errors when specified RamaLama container is missing"
    )
    parser.add_argument("NAME", nargs="?", completer=local_containers)
    parser.set_defaults(func=stop_container)


def stop_container(args):
    from ramalama.runtime import engine

    if not args.all:
        engine.stop_container(args, args.NAME)
        return

    if args.NAME:
        raise ValueError(f"specifying --all and container name, {args.NAME}, not allowed")
    args.ignore = True
    args.format = "{{ .Names }}"
    for i in engine.containers(args):
        engine.stop_container(args, i)
