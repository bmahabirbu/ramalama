import tempfile
import os
import json
import logging
import subprocess
import sys
from subprocess import CalledProcessError
from pathlib import Path
from typing import Iterable

import hashlib
import uuid
import shutil
import requests

from argparse import Namespace

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# New imports
from fastapi import FastAPI
import uvicorn

# we also need fastembed
from qdrant_client import QdrantClient, models

from typing import Iterator

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_core.documents import Document

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_openai.chat_models import ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

_log = logging.getLogger(__name__)

ociimage_rag = "org.containers.type=ai.image.rag"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM = ""
COLLECTION_NAME = "docs"

class Rag:
    def __init__(self, args):
        self.args = args

        logging.basicConfig(level=logging.DEBUG if self.args.debug else logging.ERROR)

        # These will eventually come from args
        self.target = self.args.IMAGE
        self.storage_file_path = os.path.dirname(os.getcwd()) + "/ramalama"

        # Always import files before starting Database class
        # We need to mount the persistant volume before starting 
        # the Qdrant container to avoid errors

        self.import_files()

        self.vector_database = Database(self.args)
        self.conv = Converter()

        self.llm_instance = "http://localhost:8080/completion"
        self.agent_llm_instance = "http://localhost:8081/completion"
    
    def init_database(self):
        self.vector_database.init_database()
    
    def configure_database(self):
        self.vector_database.configure_database()
    
    def add_files(self, file_path):
        try:
            documents, metadata, ids = self.conv.convert(file_path)
        except Exception as e:
            _log.error(f"Couldn't add files: {e}")
            return "Couldn't add files"
        self.vector_database.add(documents, metadata, ids)
        return "Added Files"
    
    def clean_up(self):
        self.vector_database.clean_up()
    
    def export_files(self):
        self.vector_database.export_database(self.target)

    def import_files(self):
        """
        Imports files from a Qdrant persistent volume image by running a container, 
        copying data to the host, and cleaning up the container.
        """

        container_name = "temp-container"
        
        try:
            # Step 1: Start the container
            try:
                run_cmd(
                    [self.args.engine, "run", "-d", "--name", container_name, self.target],
                    debug=self.args.debug,
                )
                _log.info(f"Container '{container_name}' started successfully.")
            except CalledProcessError as e:
                _log.warning("No Qdrant storage image available, attempting to pull...")
                try:
                    run_cmd(
                        [self.args.engine, "pull", self.target],
                        debug=self.args.debug,
                    )
                    _log.info("Successfully pulled storage image.")
                    run_cmd(
                        [self.args.engine, "run", "-d", "--name", container_name, self.target],
                        debug=self.args.debug,
                    )
                except CalledProcessError as pull_error:
                    _log.error("Failed to pull storage image.")
                    _log.error(pull_error)
                    return

            # Step 2: Copy the folder to the host
            try:
                run_cmd(
                    [
                        self.args.engine,
                        "cp",
                        f"{container_name}:/qdrant_storage",
                        self.storage_file_path,
                    ],
                    debug=self.args.debug,
                )
                _log.info("Data copied successfully from container to host.")
            except CalledProcessError as copy_error:
                _log.error("Failed to copy data from container to host.")
                _log.error(copy_error)
                return

        finally:
            # Step 3: Stop and remove the container
            try:
                run_cmd(
                    [self.args.engine, "stop", container_name],
                    debug=self.args.debug,
                )
                _log.info(f"Container '{container_name}' stopped.")
            except CalledProcessError:
                _log.warning(f"Container '{container_name}' might already be stopped.")

            try:
                run_cmd(
                    [self.args.engine, "rm", container_name],
                    debug=self.args.debug,
                )
                _log.info(f"Container '{container_name}' removed.")
            except CalledProcessError:
                _log.warning(f"Container '{container_name}' might already be removed.")

        _log.info("Successfully added files from Qdrant persistent volume image.")
    
    def push_files_to_cloud(self):
        pass

    def kube(self):
        pass

    def agentic_query(self, text):
        # Doesn't work quite yet 
        context = self.query_database(text)
        formatted_query = TEMPLATE_Q.format(context=context, query=text)

        answer = self.query_api(formatted_query, self.agent_llm_instance)

        _log.debug("ANSWER from agent: %s", answer)

        if answer == "1":
            formatted_query = TEMPLATE.format(context=context, query=text)
        elif answer == "0":
            formatted_query = TEMPLATE_NO_CONTEXT.format(query=text)
        else:
            _log.error("Unknown error occurred")
            return
        result = self.query_api(formatted_query, self.llm_instance)
        return result
    
    def query(self, text):
        context = self.query_database(text)
        formatted_query = TEMPLATE.format(context=context, query=text)
        result = self.query_api(formatted_query, self.llm_instance)
        return result
    
    def query_database(self, text):
        return self.vector_database.search(text)
    
    def query_api(self, formatted_query, llm_host) -> str:
        # Define the payload
        payload = {
            "prompt": formatted_query,
            "n_predict": 20,
            "no-warmup": "",
            "stop": ["\n"]
        }

        # Define the headers
        headers = {
            "Content-Type": "application/json"
        }

        # Send the POST request
        response = requests.post(llm_host, json=payload, headers=headers)

        # Log the response
        if response.status_code == 200:
            result = response.json().get("content", "")
            return result.strip()
        else:
            _log.error("Error: %s %s", response.status_code, response.text)
            return ""

    def run(self):
        print("> Welcome to the Rag Assistant!")
        try:
            while True:
                # User input
                user_input = input("> ").strip()

                # Skip empty queries
                if not user_input:
                    print("> Please enter a valid query.")
                    continue
                
                # Check for a specific query
                result = self.query(user_input)
                print("> Assistant: %s", result)

        except KeyboardInterrupt:
            print("\n> Exiting... Goodbye!")  # Catch any Interrupts and exit gracefully
            # self.clean_up()

    def serve(self):
        # FastAPI app for serving
        app = FastAPI()

        @app.get("/api/search")
        def search(user_input: str):
            result = self.query(user_input)
            return {"result": result}
        
        @app.get("/api/add_files")
        def add_files(file_path: str):
            result = self.add_files(file_path)
            return {"result": result}
            
        uvicorn.run(app, host="0.0.0.0", port=8000)

        # Probably edit clean up to push container as well to save data
        self.vector_database.clean_up()


class Database:
    def __init__(self, args):
        self.engine = args.engine
        self.debug = args.debug
        self.volume_path = args.PATH[0]

    def configure_database(self):
        # self.init_database()
        self.collection_name = COLLECTION_NAME

        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.qdrant_client.set_model(DENSE_MODEL)
        # comment this line to use dense vectors only
        # self.qdrant_client.set_sparse_model(SPARSE_MODEL)

    def init_database(self):
        print(self.volume_path)
        if self.volume_path is None:
            self.volume_path = os.path.join(os.path.dirname(os.getcwd()), "qdrant_storage")
            os.makedirs(self.volume_path, exist_ok=True)
        else:
            os.makedirs(self.volume_path, exist_ok=True)
        try:
            run_cmd(
                [self.engine, "run", "-d", "--name", "qdrant_container", "-p", "6333:6333", "-v", self.volume_path + ":/qdrant/snapshots", "docker.io/qdrant/qdrant"],
                debug=self.debug,
            )
        except Exception as e:
            _log.warning(f"Failed to initialize database container: {e}")
            self.start_database()

    def get_client(self):
        return self.qdrant_client
    
    def start_database(self):
        run_cmd(
            [self.engine, "start", "qdrant_container"],
            debug=self.debug,
        )
    
    def stop_database(self):
        run_cmd(
            [self.engine, "stop", "qdrant_container"],
            debug=self.debug,
        )
    
    def delete_database(self):
        run_cmd(
            [self.engine, "rm", "qdrant_container"],
            debug=self.debug,
        )
    
    def clear_database(self):
        """Clear the Qdrant Vector Database"""
        try:
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            _log.warning(f"Database already cleared or error encountered: {e}")
    
    def push_database(self, volume_path="", image_name=""):
        # push the database image to the cloud
        pass

    def export_database(self, target):
        _log.info(f"Building {target}...")
        # Check if the target image already exists and remove it if necessary
        try:
            existing_image_id = run_cmd(
                [self.engine, "images", "-q", target],
                debug=self.debug,
            ).stdout.decode("utf-8").strip()
            
            # If an old image exists, remove it
            if existing_image_id:
                _log.info(f"Old image with tag {target} found. Removing it...")
                run_cmd(
                    [self.engine, "rmi", existing_image_id],
                    debug=self.debug,
                )
                _log.info(f"Old image {target} removed.")
        except Exception as e:
            _log.warning(f"Error checking/removing old image: {e}")

        # Create a temporary container file for building the image
        contextdir = os.path.dirname(os.getcwd())
        containerfile = tempfile.NamedTemporaryFile(prefix='RamaLama_Containerfile_', delete=True)

        # Open the file for writing
        with open(containerfile.name, 'w') as c:
            c.write(
                f"""
    FROM registry.access.redhat.com/ubi9/ubi-micro:9.4-15
    COPY qdrant_storage/ /qdrant_storage/
    LABEL {ociimage_rag}
    """
            )

        # Build the new image
        try:
            imageid = (
                run_cmd(
                    [self.engine, "build", "-t", target, "--no-cache", "-q", "-f", containerfile.name, contextdir],
                    debug=self.debug,
                )
                .stdout.decode("utf-8")
                .strip()
            )
            _log.info(f"New image {target} built successfully.")
            return imageid
        except Exception as e:
            _log.error(f"Failed to build new image: {e}")
            return None
    
    def search(self, text: str, limit=5):
        points = self.qdrant_client.query(self.collection_name, query_text=text, limit=limit)
        context = ""
        for point in points:
            context += point.document + " "
        return context.strip()
    
    def add(self, documents, metadatas, ids):
        self.qdrant_client.add(self.collection_name, documents=documents, metadata=metadatas, ids=ids, batch_size=64)

    def info(self):
        info = self.qdrant_client.get_collection(self.collection_name)
        _log.info(f"Collection Info: {info.points_count}")

    def clean_up(self):
        _log.info("Cleaning up database...")
        if self.volume_path is None:
            _log.warning("No Database Volume Found")
        else:
            shutil.rmtree(self.volume_path)
        self.stop_database()
        self.delete_database()


class Converter:
    """A Class designed to handle all document conversions"""
    def __init__(self):
        self.doc_converter = DocumentConverter()

    def convert(self, file_path):
        targets = []
        _log.info(f"Converting files from path: {file_path}")

        # Check if file_path is a directory or a file
        if os.path.isdir(file_path):
            targets.extend(self.walk(file_path))  # Walk directory and process all files
        elif os.path.isfile(file_path):
            targets.append(file_path)  # Process the single file
        else:
            _log.error("Invalid file path provided")
            return False

        result = self.doc_converter.convert_all(targets)

        documents = []
        ids = []
        for file in result:
            # Chunk the document using HybridChunker
            for chunk in HybridChunker(tokenizer=DENSE_MODEL, max_tokens=1000).chunk(dl_doc=file.document):
                # Extract the text and metadata from the chunk
                document = Document(
                    page_content=chunk.text,
                    metadata=chunk.meta.export_json_dict(),
                )

                # Append
                documents.append(document)
                
                # Generate unique ID for the chunk
                ids.append(chunk.meta.origin.binary_hash)
        return documents, ids

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


# def generate(args):
#     _log.info("Generating Rag instance...")
#     _log.debug(f"Args provided: {args}")
    
#     rag = Rag(args)
#     rag.init_database()
#     rag.configure_database()

#     rag.add_files(file_path=args.PATH[0])
#     rag.export_files()
#     rag.clean_up()
#     # finally generate kube 

#     # print(rag.query_database("brians email address"))

def perror(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def run_cmd(args, cwd=None, stdout=subprocess.PIPE, ignore_stderr=False, debug=False):
    """
    Run the given command arguments.

    Args:
    args: command line arguments to execute in a subprocess
    cwd: optional working directory to run the command from
    """
    if debug:
        perror("run_cmd: ", *args)

    stderr = None
    if ignore_stderr:
        stderr = subprocess.PIPE

    return subprocess.run(args, check=True, cwd=cwd, stdout=stdout, stderr=stderr)

# ## TODO

# # Add the ability to call ramalama serve to start instance
# # Add kube play ability

# if __name__ == "__main__":
#     from argparse import Namespace

#     args = Namespace(
#         debug=False,
#         engine='podman',
#         PATH=[],
#         IMAGE='qstorage:latest',
#     )

#     rag = Rag(args)
#     rag.configure_database()
#     rag.run()

def import_files(args, target, storage_file_path=None):
    """
    Imports files from a Qdrant persistent volume image by running a container, 
    copying data to the host, and cleaning up the container.
    """

    if storage_file_path == None:
        storage_file_path = os.path.dirname(os.getcwd())

    container_name = "temp-container"
    
    try:
        # Step 1: Start the container
        try:
            run_cmd(
                [args.engine, "run", "-d", "--name", container_name, target],
                debug=args.debug,
            )
            _log.info(f"Container '{container_name}' started successfully.")
        except CalledProcessError as e:
            _log.warning("No Qdrant storage image available, attempting to pull...")
            try:
                run_cmd(
                    [args.engine, "pull", target],
                    debug=args.debug,
                )
                _log.info("Successfully pulled storage image.")
                run_cmd(
                    [args.engine, "run", "-d", "--name", container_name, target],
                    debug=args.debug,
                )
            except CalledProcessError as pull_error:
                _log.error("Failed to pull storage image.")
                _log.error(pull_error)
                return

        # Step 2: Copy the folder to the host
        try:
            run_cmd(
                [
                    args.engine,
                    "cp",
                    f"{container_name}:/qdrant_storage",
                    storage_file_path,
                ],
                debug=args.debug,
            )
            _log.info("Data copied successfully from container to host.")
        except CalledProcessError as copy_error:
            _log.error("Failed to copy data from container to host.")
            _log.error(copy_error)
            return

    finally:
        # Step 3: Stop and remove the container
        try:
            run_cmd(
                [args.engine, "stop", container_name],
                debug=args.debug,
            )
            _log.info(f"Container '{container_name}' stopped.")
        except CalledProcessError:
            _log.warning(f"Container '{container_name}' might already be stopped.")

        try:
            run_cmd(
                [args.engine, "rm", container_name],
                debug=args.debug,
            )
            _log.info(f"Container '{container_name}' removed.")
        except CalledProcessError:
            _log.warning(f"Container '{container_name}' might already be removed.")

    _log.info("Successfully added files from Qdrant persistent volume image.")



class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        # Check if input is a directory or list of file paths
        if isinstance(file_path, str) and os.path.isdir(file_path):
            # If it's a directory, list all PDF files in the directory
            self._file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdf')]
        elif isinstance(file_path, list):
            # If it's a list, use the list as file paths
            self._file_paths = file_path
        else:
            # Otherwise assume it's a single file path
            self._file_paths = [file_path]

        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterable[LCDocument]:
        # Iterate over all files and convert them
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

FILE_PATH = "/mnt/c/Users/bmahabir/Desktop/pdfs"  # DocLayNet paper

# Load and split documents
loader = DoclingPDFLoader(file_path=FILE_PATH)

chunker = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = loader.load()
splits = chunker.split_documents(docs)

# Use FastEmbed embeddings
embeddings = FastEmbedEmbeddings()

# Setup database
args = Namespace(
    debug=False,
    engine='podman',
    PATH=['/home/brian/ramalama/qdrant_storage'],
    IMAGE='qstorage:latest',
)

import_files(args, "qs")

database = Database(args)
database.configure_database()
database.init_database()

client = database.get_client()

# # restore data
snapshot_directory = "/home/brian/ramalama/qdrant_storage/docs"
# List all files in the directory
snapshot_files = [f for f in os.listdir(snapshot_directory) if f.endswith(".snapshot")]

# Check if there are any snapshot files
if snapshot_files:
    # Select the first snapshot file
    snapshot_file = snapshot_files[0]
    snapshot_file_path = f"file:///qdrant/snapshots/docs/{snapshot_file}"
    
    # Recover the snapshot
    client.recover_snapshot(COLLECTION_NAME, snapshot_file_path)
    print(f"Snapshot {snapshot_file} has been successfully recovered.")
else:
    print("No snapshot files found in the directory.")
# We can also do this may be useful for kube

# curl -X POST \
#   'http://localhost:6333/collections/docs/snapshots/upload?priority=snapshot' \
#   -H 'api-key: my-api-key' \
#   -H 'Content-Type: multipart/form-data' \
#   -F 'snapshot=@/home/brian/ramalama/ramalama/docs-4959314178120938-2024-12-30-07-06-08.snapshot'

# # Set up Qdrant for vector storage
QDRANT_URI = "http://localhost:6333"  # Assuming a local Qdrant instance

vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    url=QDRANT_URI,
    collection_name=COLLECTION_NAME,
)

# Add files

# Use Local AI for the LLM
llm = ChatOpenAI(temperature=0.5,
                max_tokens=None,
                openai_api_base="http://localhost:8080", 
                openai_api_key="ed")

# Format documents for retrieval
def format_docs(docs: Iterable[LCDocument]):
    return "\n\n".join(doc.page_content for doc in docs)

# Setup database as part of the pipeline
retriever = vectorstore.as_retriever()

# Set up prompt
prompt = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# call rag chain with input and stream output (interrupt to close cleanly)
try:
    for chunk in rag_chain.stream("Give me all the projects Brian worked"):
        print(chunk, end="", flush=True)
except KeyboardInterrupt:
    print("\nStream interrupted.")
print(" ")

# client = database.get_client()
# client.create_snapshot(collection_name=COLLECTION_NAME)
# print(client.list_snapshots(collection_name=COLLECTION_NAME))

database.export_database("qs")
database.clean_up()
