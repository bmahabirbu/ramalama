import tempfile
import os
import json
import logging
from pathlib import Path
from typing import Iterable

import hashlib
import uuid
import shutil
import requests

from ramalama.common import run_cmd

# New imports
from fastapi import FastAPI
import uvicorn

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

# we also need fastembed
from qdrant_client import QdrantClient


_log = logging.getLogger(__name__)

ociimage_rag = "org.containers.type=ai.image.rag"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
LLM = ""
AGENT_LLM = ""
COLLECTION_NAME = "docs"

TEMPLATE = """

    Here is the context to use to answer the question. Some of the context might not relate to the question.
    If none of the context relates to the question answer the question without the context.

    The context is, {context}

    Now, review the question and answer it:

    The question is, {query}

    Answer: 
"""

TEMPLATE_NO_CONTEXT = """
    Now, review the prompt and answer it:

    The prompt is, {query}
"""

TEMPLATE_Q = """
Your task is to determine whether the given question can be answered based on the provided context. 

Return 1 if the question can be answered with the context, and return 0 if it cannot.

Context: {context}

Question: {query}

Answer: (0 or 1 ONLY)
"""

class Rag:
    def __init__(self, args):
        self.args = args

        model = New(args.MODEL, args)
        model.serve(args)


        # These will eventually come from args
        self.target = "qstorage:latest"
        self.storage_file_path=os.path.dirname(os.getcwd())+"/ramalama"

        # Always import files before starting Database class
        # We need to mount the persistant volume before starting 
        # the Qdrant container to avoid errors
        self.import_files()

        self.vector_database = Database(self.args)
        self.conv = Converter()

        self.llm_instance = "http://localhost:8080/completion"
        self.agent_llm_instance = "http://localhost:8081/completion"
    
    def add_files(self, file_path):
        try:
            documents, metadata, ids = self.conv.convert(file_path)
        except:
            print("Couldnt add files")
            return "Couldnt add files"
        self.vector_database.add(documents, metadata, ids)
        return "Added Files"
    
    def export_files(self):
        self.vector_database.export_database(self.target)

    def import_files(self):
        # We have to run this command before starting the Database class
        # to avoid errors with resetting the api connection
        # start the container
        try:
            run_cmd(
                    [self.args.engine, "run", "-d", "--name", "temp-container", self.target],
                    debug=self.args.debug,
                )
        except:
            print("no qdrant storage image available trying to pull...")
            ## TODO
            # Add code to pull from quay.io
            print("Couldnt pull data")
            return 
        # copy the folder to the host
        run_cmd(
                [self.args.engine, "cp", "temp-container:/qdrant_storage", self.storage_file_path],
                debug=self.args.debug,
            )
        # stop the container
        run_cmd(
                [self.args.engine, "stop", "temp-container"],
                debug=self.args.debug,
            )
        # delete the container
        run_cmd(
                [self.args.engine, "rm", "temp-container"],
                debug=self.args.debug,
            )
        print("Added files from qdrant persistant volume image")
    
    def push_files_to_cloud(self):
        pass

    def kube(self):
        pass

    def agentic_query(self, text):
        # Doesnt work quite yet 
        context = self.vector_database.search(text)
        formatted_query = TEMPLATE_Q.format(context=context, query=text)

        answer = self.query_api(formatted_query, self.agent_llm_instance)

        print("Done Querying")
        print("ANSWER from agent:", answer)
        print("Done")

        if answer == "1":
            formatted_query = TEMPLATE.format(context=context, query=text)
        elif answer == "0":
            formatted_query = TEMPLATE_NO_CONTEXT.format(query=text)
        else:
            print("unknown error occured")
            return
        result = self.query_api(formatted_query, self.llm_instance)
        return result
    
    def query(self, text):
        context = self.vector_database.search(text)
        formatted_query = TEMPLATE.format(context=context, query=text)
        result = self.query_api(formatted_query, self.llm_instance)
        return result
    
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

        # Print the response
        if response.status_code == 200:
            result = response.json().get("content", "")
            return result.strip()
        else:
            print("Error:", response.status_code, response.text)
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
                print("> Assistant:", result)

        except:
            print("\n> Exiting... Goodbye!")  # Catch any Interrupts and exit gracefully
            self.vector_database.clean_up()

    def serve(self):
        # Fast app for serving
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
        self.volume_path = None

        self.init_database()
        self.collection_name = COLLECTION_NAME

        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.qdrant_client.set_model(DENSE_MODEL)
        # comment this line to use dense vectors only
        self.qdrant_client.set_sparse_model(SPARSE_MODEL)

        # names = self.qdrant_client.get_collections()
        # print(names)


    def init_database(self):
        if self.volume_path is None:
            self.volume_path = os.path.join(os.getcwd(), "qdrant_storage")
            os.makedirs(self.volume_path, exist_ok=True)
        try:
            run_cmd(
               [self.engine, "run", "-d", "--name", "qdrant_container", "-p", "6333:6333", "-v", self.volume_path+":/qdrant/storage", "docker.io/qdrant/qdrant"],
                debug=self.debug,
            )
        except:
            self.start_database()
    
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
        except:
            print("Database already cleared") 
        
    def push_database(self, volume_path="", image_name=""):
        # push the database image to the cloud
        pass

    def export_database(self, target):
        print(self.volume_path)
        print(f"Building {target}...")
        contextdir = os.path.dirname(os.getcwd())
        containerfile = tempfile.NamedTemporaryFile(prefix='RamaLama_Containerfile_', delete=True)
        # Open the file for writing.
        with open(containerfile.name, 'w') as c:
            c.write(
                f"""\
    FROM registry.access.redhat.com/ubi9/ubi-micro:9.4-15
    COPY ramalama/qdrant_storage/ /qdrant_storage/
    LABEL {ociimage_rag}
    """
            )
        imageid = (
            run_cmd(
                [self.engine, "build", "-t", target, "--no-cache", "-q", "-f", containerfile.name, contextdir],
                debug=self.debug,
            )
            .stdout.decode("utf-8")
            .strip()
        )
        return imageid
    
    def search(self, text: str, limit=5):
        points = self.qdrant_client.query(self.collection_name, query_text=text, limit=limit)
        context = ""
        for point in points:
            context += point.document + " "
        return context.strip()
    
    def add(self, documents, metadatas, ids):
         ids = self.qdrant_client.add(self.collection_name, documents=documents, metadata=metadatas, ids=ids, batch_size=64)

    def info(self):
        info = self.qdrant_client.get_collection(self.collection_name)
        print(info.points_count,"\n")

    def clean_up(self):
        print("cleaning up database")
        if self.volume_path == None:
            print("No Database Volume")
        else:
            shutil.rmtree(self.volume_path)
        self.stop_database()
        self.delete_database()

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


def generate(args):

    print(args)
    # Start a ramalama serve instance given the model name

    # in rag.add_files file_path should be the args.PATH
    
    rag = Rag(args)
    rag.add_files(file_path="/mnt/c/Users/bmahabir/Desktop/pdfs")
    # rag.export_files()

    # select whether to use run or serve
    # serve will be used for kube play
    rag.run()


## TODO

# Add the ability to call ramalama serve to start instance
# Add kube play ability
