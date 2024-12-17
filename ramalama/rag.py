import tempfile
import os
import json
import logging
from pathlib import Path
from typing import Iterable

import hashlib
import uuid
import shutil

from ramalama.common import run_cmd

# New imports
from fastapi import FastAPI
import uvicorn

from docling.chunking import HybridChunker
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

# from docling_core.transforms.chunker import HierarchicalChunker
# from docling.datamodel.base_models import InputFormat

from qdrant_client import QdrantClient


_log = logging.getLogger(__name__)

ociimage_rag = "org.containers.type=ai.image.rag"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "docs"

class Rag:
    def __init__(self):
        pass
    def query(self, text):
        # get context from data base and query rama llm for answer
        pass


class Database:
    def __init__(self, args):

        self.engine = args.engine
        self.debug = args.debug

        self.init_database()
        self.collection_name = COLLECTION_NAME

        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.qdrant_client.set_model(DENSE_MODEL)
        # comment this line to use dense vectors only
        self.qdrant_client.set_sparse_model(SPARSE_MODEL)

    def init_database(self, volume_path="/home/brian/ramalama/ramalama/qdrant_storage"):
        os.makedirs(volume_path, exist_ok=True)
        try:
            run_cmd(
               [self.engine, "run", "-d", "--name", "qdrant_container", "-p", "6333:6333", "-v", volume_path+":/qdrant/storage", "docker.io/qdrant/qdrant"],
                debug=self.debug,
            )
        except:
            print("already running")
    
    def search(self, text: str):
        points = self.qdrant_client.query(self.collection_name, query_text=text, limit=5)
        print("<=== Retrieved documents ===>")
        context = ""
        for point in points:
            context += point.document+" "
        return context
    
    def add(self, documents, metadatas, ids):
         ids = self.qdrant_client.add(self.collection_name, documents=documents, metadata=metadatas, ids=ids, batch_size=64)

    def clear(self):
        """Clear the Qdrant Vector Database"""
        self.qdrant_client.delete_collection(collection_name=self.collection_name)

    def info(self):
        info = self.qdrant_client.get_collection(self.collection_name)
        print(info.points_count,"\n")

    def clean_up(self, volume_path="/home/brian/ramalama/ramalama/qdrant_storage"):
        shutil.rmtree(volume_path)
        run_cmd(
               [self.engine, "stop", "qdrant_container"],
                debug=self.debug,
            )
        run_cmd(
               [self.engine, "rm", "qdrant_container"],
                debug=self.debug,
            )


class Converter:
    """A Class desgined to handle all document conversions"""
    def __init__(self):
        self.doc_converter = DocumentConverter()

    def convert(self, file_path="/mnt/c/Users/bmahabir/Desktop/pdfs"):
        targets = []

        # Check if file_path is a directory or a file
        if os.path.isdir(file_path):
            targets.extend(self.walk(file_path))  # Walk directory and process all files
        elif os.path.isfile(file_path):
            targets.append(file_path)  # Process the single file

        result = self.doc_converter.convert_all(targets)

        documents, metadatas, ids = [], [], []
        for file in result:
            # Chunk the document using HybridChunker
            for chunk in HybridChunker(tokenizer=DENSE_MODEL, max_tokens=64).chunk(dl_doc=file.document):
                # Extract the text and metadata from the chunk
                doc_text = chunk.text
                doc_meta = chunk.meta.export_json_dict() 

                # Append to respective lists
                documents.append(doc_text)
                metadatas.append(doc_meta)
                
                # Generate unique ID for the chunk
                doc_id = self.generate_hash(doc_text)
                ids.append(doc_id)
        return documents, metadatas, ids

    def walk(self, path):
        targets = []
        for root, dirs, files in os.walk(path, topdown=True):
            if len(files) == 0:
                continue
            for f in files:
                file = os.path.join(root, f)
                if os.path.isfile(file):
                    targets.append(file)
        return targets
    
    def generate_hash(self, document: str) -> str:
        """Generate a unique hash for a document."""
            # Generate SHA256 hash of the document text
        sha256_hash = hashlib.sha256(document.encode('utf-8')).hexdigest()
        
        # Use the first 32 characters of the hash to create a UUID
        return str(uuid.UUID(sha256_hash[:32]))



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

    vector_database = Database(args)
    conv = Converter()
    documents, metadata, ids = conv.convert()
    vector_database.add(documents, metadata, ids)

    # Propbably can create a seperate function
    # for "run" capabilites in a while loop handling queries
    # print(vector_database.search("Sol Clarity"))
    # vector_database.clean_up()

    # Fast app for serving
    app = FastAPI()

    @app.get("/api/search")
    def search(q: str):
        return {"result": vector_database.search(text=q)}
        
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Probably edit clean up to push container as well to save data
    vector_database.clean_up()

## TODO
# Add functions to clear data base remove files and update files
# Add a template so that the ai knows not to use the context if it doesnt pertain to the questions
# On that note add the ability to further inspect scores and context so the context relates properly to the question
# Fix how docling chunks data so that in properly incorporates data
# if for example a chunk in a data is really high give more of that data
# Add the ability for more verbose rag with more context

# Add the ability to run locally or using kube play (VERY IMPORTANT)
# Add fastapi for kube play