import tempfile
import os
import json
import logging
import subprocess
import sys
from subprocess import CalledProcessError
from pathlib import Path
from typing import Iterable

import time

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
from qdrant_client.http.models import Distance, VectorParams

from typing import Iterator

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_core.documents import Document

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_openai.chat_models import ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

_log = logging.getLogger(__name__)

ociimage_rag = "org.containers.type=ai.image.rag"

# QDRANT_URL = "http://localhost:6333"
QDRANT_URL = "http://0.0.0.0:6333"
MODEL_URL = "http://0.0.0.0:8080"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM = ""
COLLECTION_NAME = "docs"

class Rag:
    def __init__(self, args):
        self.args = args

        logging.basicConfig(level=logging.DEBUG if self.args.debug else logging.ERROR)

        self.target = self.args.IMAGE

        self.database = Database(self.args)
        self.conv = Converter()

    def start_database(self):
        self.database.start_database()
        
    def add_files(self, file_path):
        documents, metadata, ids = self.conv.convert(file_path)
        self.vector_store = self.database.add_files(documents, metadata, ids)
    
    def restore(self):
        self.vector_store = self.database.restore_database(self.target)
    
    def restore_kube(self):
         self.vector_store = self.database.restore_database_kube()
    
    def clean_up(self):
        self.database.clean_up()
    
    def export_files(self):
        self.database.export_database(self.target)
        # TODO
        # Push to cloud
     
    def format_docs(self, docs: Iterable[LCDocument]):
        # Format documents for retrieval
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query_database(self, text):
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        retrieved_docs = retriever.invoke("Brian's real name")

        context = self.format_docs(retrieved_docs)

    
    def create_chain(self):
        # Create interface to talk with ramalama served llm
        llm = ChatOpenAI(temperature=0.5,
                max_tokens=None,
                openai_api_base=MODEL_URL, 
                openai_api_key="ed")


        # Setup database as part of the pipeline
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})

        # Set up prompt
        prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
        )

        # Create RAG chain
        self.rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def chain_from_scratch(self, text):
        # Step 1: Start and End time for document retrieval
        start_time = time.time()
        
        # Step 2: Set up retriever to fetch relevant documents
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        
        # Step 3: Retrieve documents based on the query
        retrieved_docs = retriever.invoke("Brian's real name")
        
        # End time for retrieval
        end_time = time.time()
        print(f"Document retrieval time: {end_time - start_time:.4f} seconds")
        
        # Step 4: Start and End time for document formatting
        start_time = time.time()
        context = self.format_docs(retrieved_docs)
        
        # End time for formatting docs
        end_time = time.time()
        print(f"Document formatting time: {end_time - start_time:.4f} seconds")
        
        # Step 5: Set up the LLM (using the local server)
        llm = ChatOpenAI(temperature=0.5,
                        max_tokens=None,
                        openai_api_base="http://localhost:8080", 
                        openai_api_key="ed")
        
        # Step 6: Set up the prompt template to structure the input for the model
        prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
        )

        # Step 7: Start and End time for preparing the prompt
        start_time = time.time()
        final_prompt = prompt.format(context=context, question=text)

        print("Size of final prompt: ", len(final_prompt))
        
        # End time for preparing the prompt
        end_time = time.time()
        print(f"Time to prepare final prompt: {end_time - start_time:.4f} seconds")
        
        # Step 8: Start and End time for invoking the LLM
        start_time = time.time()

        for chunk in llm.stream(final_prompt):
            print(chunk.content, end="", flush=True)
        print(" ")

        
        # End time for LLM invocation
        end_time = time.time()
        print(f"LLM invocation time: {end_time - start_time:.4f} seconds")

    def kube(self):
        pass

    def run(self):
        print("> Welcome to the Rag Assistant!")
        try:
            self.create_chain()
            while True:
                # User input
                user_input = input("> ").strip()

                # Skip empty queries
                if not user_input:
                    print("> Please enter a valid query.")
                    continue
                
                # Check for a specific query
                try:
                    for chunk in self.rag_chain.stream(user_input):
                        print(chunk, end="", flush=True)
                except KeyboardInterrupt:
                    print("\nStream interrupted.")
                print(" ")

        except KeyboardInterrupt:
            print("\n> Exiting... Goodbye!")  # Catch any Interrupts and exit gracefully
            self.clean_up()

    def serve(self):
        # FastAPI app for serving
        app = FastAPI()

        @app.get("/api/search")
        def search(user_input: str):
            result = self.rag_chain.invoke(user_input)
            return {"result": result}
        
        @app.get("/api/add_files")
        def add_files(file_path: str):
            result = self.add_files(file_path)
            return {"result": result}
            
        uvicorn.run(app, host="0.0.0.0", port=8000)


class Database:
    def __init__(self, args):
        self.engine = args.engine
        self.debug = args.debug
        self.volume_path = None

        if self.volume_path is None:
            self.volume_path = os.path.join(os.path.dirname(os.getcwd()), "qdrant_storage")
            os.makedirs(self.volume_path, exist_ok=True)
        else:
            os.makedirs(self.volume_path, exist_ok=True)

        # if image is passed try to restore data before starting client

        # self.init_database()
        self.collection_name = COLLECTION_NAME

        # initialize Qdrant client
        self.qdrant_client = QdrantClient(path=self.volume_path)

        # Configure embedding model
        self.embeddings = FastEmbedEmbeddings()
    
    def restore_database(self, target):
        if self.import_files(target) == True:
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            return vector_store
        else:
            _log.warning(f"Failed to initialize Vector Database")

    
    def add_files(self, documents, metadata, ids):
        self.add(documents, metadata, ids)
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        return vector_store

    def get_client(self):
        return self.qdrant_client
    
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
        # Create snapshot
        _log.info(f"Creating snapshot...")
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
        
    def import_files_kube(self):
        # # restore data
        print("\n")
        print(os.getcwd())
        print(" ".join(os.listdir(os.getcwd())))
        snapshot_directory = "shared/docs"
        # List all files in the directory
        snapshot_files = [f for f in os.listdir(snapshot_directory) if f.endswith(".snapshot")]
        # Check if there are any snapshot files
        if snapshot_files:
            # Select the first snapshot file
            snapshot_file = snapshot_files[0]
            snapshot_file_path = f"file:///qdrant/snapshots/docs/{snapshot_file}"

            print(snapshot_file_path)
            
            # Recover the snapshot
            self.qdrant_client.recover_snapshot(self.collection_name, snapshot_file_path)
            print(f"Snapshot {snapshot_file} has been successfully recovered.")
            return True
        else:
            print("No snapshot files found in the directory.")
        return False

        
    def import_files(self, target, storage_file_path=None):
        """
        Imports files from a Qdrant persistent volume image by running a container, 
        copying data to the host, and cleaning up the container.
        Returns True if successful, False otherwise.
        """

        if storage_file_path is None:
            storage_file_path = os.getcwd()

        container_name = "temp-container"
        success = False  # Initialize success flag

        try:
            # Step 1: Start the container
            try:
                run_cmd(
                    [self.engine, "run", "-d", "--name", container_name, target],
                    debug=self.debug,
                )
                _log.info(f"Container '{container_name}' started successfully.")
            except CalledProcessError as e:
                _log.warning("No Qdrant storage image available, attempting to pull...")
                try:
                    run_cmd(
                        [self.engine, "pull", target],
                        debug=self.debug,
                    )
                    _log.info("Successfully pulled storage image.")
                    run_cmd(
                        [self.engine, "run", "-d", "--name", container_name, target],
                        debug=self.debug,
                    )
                except CalledProcessError as pull_error:
                    _log.error("Failed to pull storage image.")
                    _log.error(pull_error)
                    return False  # Return False if pulling fails

            # Step 2: Copy the folder to the host
            try:
                run_cmd(
                    [
                        self.engine,
                        "cp",
                        f"{container_name}:/qdrant_storage",
                        storage_file_path,
                    ],
                    debug=self.debug,
                )
                _log.info("Data copied successfully from container to host.")
            except CalledProcessError as copy_error:
                _log.error("Failed to copy data from container to host.")
                _log.error(copy_error)
                return False  # Return False if copying fails

            success = True  # Set success flag to True if no errors encountered

        finally:
            # Step 3: Stop and remove the container
            try:
                run_cmd(
                    [self.engine, "stop", container_name],
                    debug=self.debug,
                )
                _log.info(f"Container '{container_name}' stopped.")
            except CalledProcessError:
                _log.warning(f"Container '{container_name}' might already be stopped.")

            try:
                run_cmd(
                    [self.engine, "rm", container_name],
                    debug=self.debug,
                )
                _log.info(f"Container '{container_name}' removed.")
            except CalledProcessError:
                _log.warning(f"Container '{container_name}' might already be removed.")

        if success:
            _log.info("Successfully added files from Qdrant persistent volume image.")
            return True
        else:
            _log.error("Failed to import files from Qdrant persistent volume image.")
            return False
    
    def search(self, text: str, limit=5):
        points = self.qdrant_client.query(self.collection_name, query_text=text, limit=limit)
        context = ""
        for point in points:
            context += point.document + " "
        return context.strip()
    
    def add(self, documents, metadata, ids):
        self.qdrant_client.add(self.collection_name, documents=documents, metadata=metadata, ids=ids)

    def info(self):
        info = self.qdrant_client.get_collection(self.collection_name)
        _log.info(f"Collection Info: {info.points_count}")

    def clean_up(self):
        _log.info("Cleaning up database...")
        if self.volume_path is None:
            _log.warning("No Database Mount Volume Found")
        else:
            shutil.rmtree(self.volume_path)

class Converter:
    """A Class desgined to handle all document conversions"""
    def __init__(self):
        self.doc_converter = DocumentConverter()

    def convert(self, file_path):
        targets = []

        # Check if file_path is a directory or a file
        if os.path.isdir(file_path):
            targets.extend(self.walk(file_path))  # Walk directory and process all files
        elif os.path.isfile(file_path):
            targets.append(file_path)  # Process the single file
        else:
            # if the path provided is wrong just return false
            # Used in Rag.add files to avoid errors
            return False

        result = self.doc_converter.convert_all(targets)

        documents, metadatas, ids = [], [], []
        for file in result:
            # Chunk the document using HybridChunker
            for chunk in HybridChunker(tokenizer=DENSE_MODEL, max_tokens=500).chunk(dl_doc=file.document):
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

# Imports from ramalama figuring out how to better manage this
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


def generate(args):
    # Testing Command
    # ./bin/ramalama rag /mnt/c/Users/bmahabir/Desktop/pdfs docker.io/brianmahabir/qs:latest

    # FILE_PATH = "/mnt/c/Users/bmahabir/Desktop/pdfs" 
    _log.info("Generating Rag instance...")
    _log.debug(f"Args provided: {args}")
    
    rag = Rag(args)
    rag.restore()
    rag.run()
    # rag.add_files(file_path=args.PATH[0])
    # rag.export_files()

    # Push to cloud
    # rag.push()

    # rag.clean_up()

# ## TODO

# # Add the ability to call ramalama serve to start instance
# # Add kube play ability

if __name__ == "__main__":
    FILE_PATH = "/mnt/c/Users/bmahabir/Desktop/pdfs" 
    args = Namespace(
        debug=False,
        engine='podman',
        PATH=[''],
        IMAGE='docker.io/brianmahabir/qs:latest',
    )
    print("Starting RAG in Kube Play")
    rag = Rag(args)
    # rag.restore_kube()
    rag.add_files(FILE_PATH)

    # rag.chain_from_scratch("what is brians email")

    # rag.create_chain()
    # rag.serve()

    # rag.export_files()

    # rag.clean_up()