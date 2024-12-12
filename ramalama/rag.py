import tempfile
import os
import json
import logging
from pathlib import Path
from typing import Iterable

from ramalama.common import run_cmd
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.milvus import MilvusVectorStore

_log = logging.getLogger(__name__)

ociimage_rag = "org.containers.type=ai.image.rag"


def walk(path):
    targets = []
    for root, dirs, files in os.walk(path, topdown=True):
        if len(files) == 0:
            continue
        for f in files:
            file = os.path.join(root, f)
            if os.path.isfile(file):
                targets.append(file)
    return targets


def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            # Export Docling document format to JSON:
            with (output_dir / f"{doc_filename}.json").open("w") as fp:
                fp.write(json.dumps(conv_res.document.export_to_dict()))

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(f"Document {conv_res.input.file} was partially converted with the following errors:")
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count


def build(source, target, args):
    print(f"Building {target}...")
    src = os.path.realpath(source)
    contextdir = os.path.dirname(src)
    model = os.path.basename(src)
    containerfile = tempfile.NamedTemporaryFile(prefix='RamaLama_Containerfile_', delete=True)
    # Open the file for writing.
    with open(containerfile.name, 'w') as c:
        c.write(
            f"""\
FROM scratch
COPY {model} /
LABEL {ociimage_rag}
"""
        )
    imageid = (
        run_cmd(
            [args.engine, "build", "-t", target, "--no-cache", "-q", "-f", containerfile.name, contextdir],
            debug=args.debug,
        )
        .stdout.decode("utf-8")
        .strip()
    )
    return imageid


def generate(args):
    tmpdir = tempfile.TemporaryDirectory(prefix="ramalama_", delete=True)
    targets = []
    for p in args.PATH:
        if os.path.isfile(p):
            targets.append(p)  # Process selected file
            continue
        if os.path.isdir(p):
            targets.extend(walk(p))  # Walk directory and process all files
            continue
        targets.append(p)  # WEB?

    converter = DocumentConverter()
    conv_results = converter.convert_all(targets, raises_on_error=False)
    success_count, partial_success_count, failure_count = export_documents(conv_results, output_dir=Path(tmpdir.name))
    if failure_count > 0:
        raise RuntimeError(f"failed to convert {failure_count} target(s) out of {len(targets)} documents.")

    build(tmpdir.name, args.IMAGE, args)

    # Initialize LlamaIndex components for RAG
    reader = DoclingNodeParser()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    gen_model = HuggingFaceInferenceAPI(
        token=os.getenv("HF_TOKEN"),
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    vector_store = MilvusVectorStore(
        uri=str(Path(tmpdir.name) / "docling.db"),
        dim=len(embed_model.get_text_embedding("test")),
        overwrite=True,
    )

    index = VectorStoreIndex.from_documents(
        documents=reader.load_data(targets),
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=embed_model,
    )

    query = "Your custom query here"
    result = index.as_query_engine(llm=gen_model).query(query)

    print(f"Query: {query}\nResponse: {result.response.strip()}\n")
    print("Sources:")
    print([(node.text, node.metadata) for node in result.source_nodes])

# The Plan

# First step 
# find a method to create and populate a 
# vector database locally with embedded info
# Run that database inside a container

# Second step
# Connect to the database and use Rag