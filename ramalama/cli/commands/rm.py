import subprocess

from ramalama.cli._utils import local_models, suppressCompleter, get_shortnames
from ramalama.utils.common import perror
from ramalama.model_store.global_store import GlobalModelStore
from ramalama.transports.base import MODEL_TYPES
from ramalama.transports.transport_factory import New, TransportFactory


def register(subparsers):
    parser = subparsers.add_parser("rm", help="remove AI Model from local storage")
    parser.add_argument("-a", "--all", action="store_true", help="remove all local Models")
    parser.add_argument("--ignore", action="store_true", help="ignore errors when specified Model does not exist")
    parser.add_argument("MODEL", nargs="*", completer=local_models)
    parser.set_defaults(func=rm_cli)


def _rm_oci_model(model, args) -> bool:
    try:
        m = TransportFactory(model, args, ignore_stderr=True).create_oci()
        return m.remove(args)
    except Exception:
        return False


def _rm_model(models, args):
    exceptions = []
    shortnames = get_shortnames()

    for model in models:
        model = shortnames.resolve(model)
        try:
            m = New(model, args)
            if m.remove(args):
                continue
            if args.ignore:
                _rm_oci_model(model, args)
                continue
        except (KeyError, subprocess.CalledProcessError) as e:
            for prefix in MODEL_TYPES:
                if model.startswith(prefix + "://"):
                    if not args.ignore:
                        raise e
            if _rm_oci_model(model, args) or args.ignore:
                continue
            exceptions.append(e)

    if len(exceptions) > 0:
        for exception in exceptions[1:]:
            perror("Error: " + str(exception).strip("'\""))
        raise exceptions[0]


def rm_cli(args):
    if not args.all:
        if len(args.MODEL) == 0:
            raise IndexError("one MODEL or --all must be specified")
        return _rm_model(args.MODEL, args)

    if len(args.MODEL) > 0:
        raise IndexError("can not specify --all as well MODEL")

    models = GlobalModelStore(args.store).list_models(engine=args.engine, show_container=args.container)

    failed_models = []
    for model in models.keys():
        try:
            _rm_model([model], args)
        except Exception as e:
            failed_models.append((model, str(e)))
    if failed_models:
        for model, error in failed_models:
            perror(f"Failed to remove model '{model}': {error}")
        failed_names = ', '.join([model for model, _ in failed_models])
        raise Exception(f"Failed to remove the following models: {failed_names}")
