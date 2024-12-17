import tempfile
import os
import json
import logging
from pathlib import Path
from typing import Iterable
from fastapi import FastAPI
import hashlib
import uuid
from docling.chunking import HybridChunker

from ramalama.common import run_cmd
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from docling_core.transforms.chunker import HierarchicalChunker
from qdrant_client import QdrantClient
from docling.datamodel.base_models import InputFormat

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
    # tmpdir = tempfile.TemporaryDirectory(prefix="ramalama_", delete=True)
    # targets = []
    # for p in args.PATH:
    #     if os.path.isfile(p):
    #         targets.append(p)  # Process selected file
    #         continue
    #     if os.path.isdir(p):
    #         targets.extend(walk(p))  # Walk directory and process all files
    #         continue
    #     targets.append(p)  # WEB?

    # converter = DocumentConverter()
    # conv_results = converter.convert_all(targets, raises_on_error=False)
    # success_count, partial_success_count, failure_count = export_documents(conv_results, output_dir=Path(tmpdir.name))
    # if failure_count > 0:
    #     raise RuntimeError(f"failed to convert {failure_count} target(s) out of {len(targets)} documents.")

    # build(tmpdir.name, args.IMAGE, args)


    # Set models
    model = "sentence-transformers/all-MiniLM-L6-v2"
    # For Hybird search set sparse model
    spare_model = "Qdrant/bm25"

    COLLECTION_NAME = "docling"

    doc_converter = DocumentConverter()
    # uncomment to run database locally
    # client = QdrantClient(location=":memory:")
    client = QdrantClient(location="http://localhost:6333")

    client.set_model(model)
    client.set_sparse_model(spare_model)

    # if not client.collection_exists(COLLECTION_NAME):

    file_path = "/mnt/c/Users/bmahabir/Desktop/pdfs"

    targets = []

    # Check if file_path is a directory or a file
    if os.path.isdir(file_path):
        targets.extend(walk(file_path))  # Walk directory and process all files
    elif os.path.isfile(file_path):
        targets.append(file_path)  # Process the single file

    print(targets)

    result = doc_converter.convert_all(targets)

    documents, metadatas, ids = [], [], []
    for file in result:
        # Chunk the document using HybridChunker
        for chunk in HybridChunker(tokenizer=model, max_tokens=64).chunk(dl_doc=file.document):
            # Extract the text and metadata from the chunk
            doc_text = chunk.text
            doc_meta = chunk.meta.export_json_dict() 

            # Append to respective lists
            documents.append(doc_text)
            metadatas.append(doc_meta)
            
            # Generate unique ID for the chunk
            doc_id = generate_hash(doc_text)
            ids.append(doc_id)

            
    ids = client.add(COLLECTION_NAME, documents=documents, metadata=metadatas, ids=ids, batch_size=64)

    print(ids)

    info = client.get_collection(COLLECTION_NAME)
    print(info.points_count,"\n")


    points = client.query(COLLECTION_NAME, query_text="What happens in the House of Usher", limit=10)

    print("<=== Retrieved documents ===>")
    for point in points:
        print(point.document, " Score: ", point.score, "\n")
        # print(point.metadata)

def generate_hash(document: str) -> str:
    """Generate a unique hash for a document."""
        # Generate SHA256 hash of the document text
    sha256_hash = hashlib.sha256(document.encode('utf-8')).hexdigest()
    
    # Use the first 32 characters of the hash to create a UUID
    return str(uuid.UUID(sha256_hash[:32]))

## TODO
# Add functions to clear data base remove files and update files
# Add a template so that the ai knows not to use the context if it doesnt pertain to the questions
# On that note add the ability to further inspect scores and context so the context relates properly to the question
# Fix how docling chunks data so that in properly incorporates data
# if for example a chunk in a data is really high give more of that data
# Add the ability for more verbose rag with more context

# Add the ability to run locally or using kube play (VERY IMPORTANT)
# Add fastapi for kube play