from datetime import datetime, timezone

from ramalama.cli._utils import (
    LIST_SORT_FIELD_OPTIONS,
    LIST_SORT_ORDER_OPTIONS,
    human_duration,
    human_readable_size,
)
from ramalama.model_store.global_store import GlobalModelStore
from ramalama.transports.base import trim_model_name


def register(subparsers):
    parser = subparsers.add_parser(
        "list", aliases=["ls"], help="list all downloaded AI Models (excluding partially downloaded ones)"
    )
    parser.add_argument("--all", dest="all", action="store_true", help="include partially downloaded AI Models")
    parser.add_argument("--json", dest="json", action="store_true", help="print using json")
    parser.add_argument("-n", "--noheading", dest="noheading", action="store_true", help="do not display heading")
    parser.add_argument(
        "--sort",
        dest="sort",
        choices=LIST_SORT_FIELD_OPTIONS,
        default="name",
        help="field used to sort the AI Models",
    )
    parser.add_argument(
        "--order",
        dest="order",
        choices=LIST_SORT_ORDER_OPTIONS,
        default="desc",
        help="order used to sort the AI Models",
    )
    parser.set_defaults(func=list_cli)


def _list_models_from_store(args):
    models = GlobalModelStore(args.store).list_models(engine=args.engine, show_container=args.container)

    ret = []
    local_timezone = datetime.now().astimezone().tzinfo

    for model, files in models.items():
        is_partially_downloaded = any(file.is_partial for file in files)
        if not args.all and is_partially_downloaded:
            continue

        model = trim_model_name(model)
        size_sum = 0
        last_modified = 0.0
        for file in files:
            size_sum += file.size
            last_modified = max(file.modified, last_modified)

        ret.append({
            "name": f"{model} (partial)" if is_partially_downloaded else model,
            "modified": datetime.fromtimestamp(last_modified, tz=local_timezone).isoformat(),
            "size": size_sum,
        })

    ret.sort(key=lambda entry: entry[args.sort], reverse=args.order == "desc")
    return ret


def _list_models(args):
    return _list_models_from_store(args)


def list_cli(args):
    import json

    models = _list_models(args)

    if args.json:
        print(json.dumps(models))
        return

    name_width = len("NAME")
    modified_width = len("MODIFIED")
    size_width = len("SIZE")
    for model in sorted(models, key=lambda d: d['name']):
        try:
            delta = int(datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(model["modified"]).timestamp())
            modified = human_duration(delta) + " ago"
            model["modified"] = modified
        except TypeError:
            pass
        model["size"] = human_readable_size(model["size"])
        name_width = max(name_width, len(model["name"]))
        modified_width = max(modified_width, len(model["modified"]))
        size_width = max(size_width, len(model["size"]))

    if not args.quiet and not args.noheading and not args.json:
        print(f"{'NAME':<{name_width}} {'MODIFIED':<{modified_width}} {'SIZE':<{size_width}}")

    for model in models:
        if args.quiet:
            print(model["name"])
        else:
            modified = model['modified']
            print(f"{model['name']:<{name_width}} {modified:<{modified_width}} {model['size'].upper():<{size_width}}")
