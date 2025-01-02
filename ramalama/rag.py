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

    def start_database(self):
        self.database.start_database()
        
    def add_files(self, file_path):
        converter = DoclingPDFLoader(file_path)
        splits = converter.load_and_split()
        self.vector_store = self.database.add_files(documents=splits)
    
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

        self.url = QDRANT_URL

        # self.init_database()
        self.collection_name = COLLECTION_NAME

        # initialize Qdrant client
        self.qdrant_client = QdrantClient(QDRANT_URL)

        # Configure embedding model
        self.embeddings = FastEmbedEmbeddings()

        if self.volume_path is None:
            self.volume_path = os.path.join(os.path.dirname(os.getcwd()), "qdrant_storage")
            os.makedirs(self.volume_path, exist_ok=True)
        else:
            os.makedirs(self.volume_path, exist_ok=True)

    def start_database(self):

        try:
            run_cmd(
                [self.engine, "run", "-d", "--name", "qdrant_container", "-p", "6333:6333", "-v", self.volume_path + ":/qdrant/snapshots", "docker.io/qdrant/qdrant"],
                debug=self.debug,
            )
        except Exception as e:
            _log.warning(f"Failed to initialize database container: {e}")
            run_cmd(
                [self.engine, "start", "qdrant_container"],
                debug=self.debug,
            )

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
    
    def restore_database_kube(self):
        if self.import_files_kube() == True:
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            return vector_store
        else:
            print("Cant Restore Vector Database")
            _log.warning(f"Failed to initialize Vector Database")

        
    
    def add_files(self, documents):
        vector_store = QdrantVectorStore.from_documents(
            documents,
            self.embeddings,
            url=self.url,
            collection_name=self.collection_name,
        )
        return vector_store

    def get_client(self):
        return self.qdrant_client
    
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
        # Create snapshot
        _log.info(f"Creating snapshot...")
        self.qdrant_client.create_snapshot(collection_name=self.collection_name)
        _log.info(self.qdrant_client.list_snapshots(collection_name=self.collection_name))

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
        """

        if storage_file_path == None:
            storage_file_path = os.path.dirname(os.getcwd())

        container_name = "temp-container"
        
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
                    return

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
                return

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

        _log.info("Successfully added files from Qdrant persistent volume image.")

        # # restore data
        snapshot_directory = storage_file_path + "/qdrant_storage/docs"
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

        # In a kube enviorment 

        # we will need to change this snapshot_file_path = f"file:///qdrant/snapshots/docs/{snapshot_file}
        # To a url 
    
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
            _log.warning("No Database Mount Volume Found")
        else:
            shutil.rmtree(self.volume_path)
        self.stop_database()
        self.delete_database()

class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: str | list[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Initialize the loader with a file path or a list of file paths.
        
        Args:
            file_path (str | list[str]): Path to the directory or list of PDF file paths.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap between chunks.
        """
        # Check if input is a directory or list of file paths
        if isinstance(file_path, str):
            if os.path.isdir(file_path):
                # If it's a directory, list all PDF files in the directory
                self._file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdf')]
                if not self._file_paths:
                    raise ValueError(f"No PDF files found in the directory: {file_path}")
            else:
                raise ValueError(f"The provided path is not a valid directory: {file_path}")
        elif isinstance(file_path, list):
            # If it's a list, ensure it's not empty and contains valid file paths
            if not file_path:
                raise ValueError("The provided list of file paths is empty.")
            self._file_paths = file_path
        else:
            # Otherwise assume it's a single file path and check its validity
            if not os.path.isfile(file_path) or not file_path.endswith('.pdf'):
                raise ValueError(f"The provided file path is not a valid PDF file: {file_path}")
            self._file_paths = [file_path]

        self._converter = DocumentConverter()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def lazy_load(self) -> Iterable[LCDocument]:
        """Iterates over files and converts them into LCDocument objects."""
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

    def load_and_split(self):
        """
        Load the documents and split them into chunks.
        
        Returns:
            List[LCDocument]: List of split documents.
        """
        # Load documents
        docs = self.load()
        # Split documents
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        return chunker.split_documents(docs)

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
    rag.start_database()
    rag.add_files(file_path=args.PATH[0])
    rag.export_files()

    # Push to cloud
    # rag.push()

    rag.clean_up()

    # finally generate kube with the name of the target
    # rag.kube()

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
    rag.restore_kube()
    # rag.add_files(FILE_PATH)

    # rag.chain_from_scratch("what is brians email")

    rag.create_chain()
    rag.serve()

    # rag.export_files()

    # rag.clean_up()