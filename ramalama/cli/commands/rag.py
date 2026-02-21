"""ramalama rag – convert documents to a RAG vector database container image."""

from ramalama.cli._parser import OverrideDefaultAction, default_image
from ramalama.cli._utils import default_threads, suppressCompleter
from ramalama.config import get_config


def register(subparsers):
    config = get_config()
    parser = subparsers.add_parser(
        "rag",
        help="convert documents to a RAG vector database and package as a container image",
    )
    parser.add_argument(
        "DOCUMENTS",
        help="path to document images (file or directory) to process",
    )
    parser.add_argument(
        "IMAGE_NAME",
        help="name for the output container image containing the Qdrant vector database",
    )
    parser.add_argument(
        "--docling-model",
        dest="docling_model",
        default="hf://ibm-granite/granite-docling-258M-GGUF",
        help="Granite Docling GGUF model for document conversion",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "--image",
        default=default_image(),
        help="OCI container image to use for the llama.cpp inference server",
        action=OverrideDefaultAction,
    )
    parser.add_argument(
        "--ngl",
        dest="ngl",
        type=int,
        default=config.ngl,
        help="number of layers to offload to the GPU",
        completer=suppressCompleter,
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=default_threads(),
        help="number of CPU threads to use",
        completer=suppressCompleter,
    )
    parser.set_defaults(func=rag_cli)


def rag_cli(args):
    from ramalama.rag.pipeline import run_rag_pipeline

    run_rag_pipeline(args)
