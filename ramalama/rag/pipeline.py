"""RAG pipeline: convert documents -> chunk -> embed -> store in Qdrant -> package as container image."""

import argparse
import os
import subprocess
import tempfile
import time
from http.client import HTTPConnection, HTTPException
from pathlib import Path

from ramalama.cli._parser import default_image
from ramalama.cli._utils import assemble_command_lazy, default_threads
from ramalama.config import get_config
from ramalama.transports.base import compute_serving_port
from ramalama.transports.transport_factory import New
from ramalama.runtime.engine import BuildEngine
from ramalama.utils.common import perror, set_accel_env_vars
from ramalama.utils.logger import logger

IMAGE_PARSER_MODEL = "hf://ibm-granite/granite-docling-258M-GGUF"
EMBEDDING_MODEL = "hf://Qwen/Qwen3-Embedding-0.6B-GGUF"


def run_rag_pipeline(args):
    """Execute the full RAG pipeline.

    1. Pull/verify the Granite Docling GGUF model.
    2. Pull/verify the Qwen3 Embedding GGUF model.
    3. Start temporary llama.cpp servers for document conversion and embeddings.
    4. Send each document image through the VLM to get structured text.
    5. Chunk the text and embed via llama.cpp into a Qdrant vector database.
    6. Package the Qdrant database as a container image.
    """
    from ramalama.rag.chunker import chunk_documents
    from ramalama.rag.converter import GraniteDoclingConverter
    from ramalama.rag.vectordb import LlamaCppEmbedder, store_in_qdrant

    source_path = Path(args.DOCUMENTS)
    image_name = args.IMAGE_NAME
    docling_model = getattr(args, "docling_model", IMAGE_PARSER_MODEL)
    embedding_model = getattr(args, "embedding_model", EMBEDDING_MODEL)

    if not source_path.exists():
        raise FileNotFoundError(f"Document path not found: {source_path}")

    if args.engine is None:
        raise ValueError("A container engine (podman or docker) is required to build the RAG image.")

    docling_serve_args = _build_serve_args(args, docling_model)
    docling_port = int(docling_serve_args.port)

    embed_serve_args = _build_embed_serve_args(args, embedding_model, exclude_ports=[docling_serve_args.port])
    embed_port = int(embed_serve_args.port)

    perror("Pulling Granite Docling model...")
    docling_transport = New(docling_model, docling_serve_args)
    docling_transport.ensure_model_exists(docling_serve_args)

    perror("Pulling Qwen3 Embedding model...")
    embed_transport = New(embedding_model, embed_serve_args)
    embed_transport.ensure_model_exists(embed_serve_args)

    set_accel_env_vars()

    docling_cmd = assemble_command_lazy(docling_serve_args)
    embed_cmd = assemble_command_lazy(embed_serve_args)

    perror("Starting Granite Docling inference server...")
    docling_proc = docling_transport.serve_nonblocking(docling_serve_args, docling_cmd)

    embed_proc = None
    try:
        _wait_for_server("localhost", docling_port)
        perror("Granite Docling server is ready.")

        perror("Starting Qwen3 Embedding server...")
        embed_proc = embed_transport.serve_nonblocking(embed_serve_args, embed_cmd)
        _wait_for_server("localhost", embed_port)
        perror("Qwen3 Embedding server is ready.")

        converter = GraniteDoclingConverter(api_url=f"http://localhost:{docling_port}")
        docs = converter.convert_directory(source_path)

        if not docs:
            raise ValueError("No documents were successfully converted")

        perror("Chunking documents ...")
        chunks, ids = chunk_documents(docs)
        perror(f"Created {len(chunks)} chunks from {len(docs)} document(s)")

        embedder = LlamaCppEmbedder(api_url=f"http://localhost:{embed_port}")

        with tempfile.TemporaryDirectory() as tmpdir:
            qdrant_dir = os.path.join(tmpdir, "qdrant_data")
            store_in_qdrant(chunks, ids, qdrant_dir, embedder)

            perror(f"Building container image '{image_name}'...")
            _build_rag_image(args, qdrant_dir, image_name)

        perror(f"RAG image '{image_name}' created successfully.")
    finally:
        _stop_server(embed_proc, embed_serve_args)
        _stop_server(docling_proc, docling_serve_args)


def _build_serve_args(args, model_name: str) -> argparse.Namespace:
    """Construct an ``argparse.Namespace`` suitable for an internal llama.cpp serve session."""
    config = get_config()

    serve_args = argparse.Namespace(
        MODEL=model_name,
        subcommand="serve",
        runtime="llama.cpp",
        container=args.container,
        engine=args.engine,
        store=args.store,
        dryrun=args.dryrun,
        debug=args.debug,
        quiet=True,
        noout=True,
        image=getattr(args, "image", default_image()),
        pull=getattr(args, "pull", config.pull),
        network=None,
        oci_runtime=None,
        selinux=False,
        nocapdrop=False,
        device=None,
        podman_keep_groups=False,
        privileged=False,
        env=[],
        detach=True,
        name=None,
        dri="on",
        host="localhost",
        port=None,
        context=8192,
        cache_reuse=256,
        ngl=getattr(args, "ngl", config.ngl),
        threads=getattr(args, "threads", default_threads()),
        temp=0.0,
        thinking=False,
        max_tokens=0,
        seed=None,
        webui="off",
        model_draft=None,
        runtime_args=["--special"],
        router_mode=False,
        models_max=4,
        generate=None,
        logfile=None,
        gguf=None,
        authfile=None,
        tlsverify=True,
        verify=config.verify,
    )

    serve_args.port = compute_serving_port(serve_args)
    return serve_args


def _build_embed_serve_args(
    args, model_name: str, exclude_ports: list[str] | None = None
) -> argparse.Namespace:
    """Construct serve args for the embedding model with ``--embedding --pooling last``."""
    config = get_config()

    serve_args = argparse.Namespace(
        MODEL=model_name,
        subcommand="serve",
        runtime="llama.cpp",
        container=args.container,
        engine=args.engine,
        store=args.store,
        dryrun=args.dryrun,
        debug=args.debug,
        quiet=True,
        noout=True,
        image=getattr(args, "image", default_image()),
        pull=getattr(args, "pull", config.pull),
        network=None,
        oci_runtime=None,
        selinux=False,
        nocapdrop=False,
        device=None,
        podman_keep_groups=False,
        privileged=False,
        env=[],
        detach=True,
        name=None,
        dri="on",
        host="localhost",
        port=None,
        context=8192,
        cache_reuse=256,
        ngl=getattr(args, "ngl", config.ngl),
        threads=getattr(args, "threads", default_threads()),
        temp=0.0,
        thinking=False,
        max_tokens=0,
        seed=None,
        webui="off",
        model_draft=None,
        runtime_args=["--embedding", "--pooling", "last"],
        router_mode=False,
        models_max=4,
        generate=None,
        logfile=None,
        gguf=None,
        authfile=None,
        tlsverify=True,
        verify=config.verify,
    )

    serve_args.port = compute_serving_port(serve_args, exclude=exclude_ports)
    return serve_args


def _wait_for_server(host: str, port: int, timeout: int = 180):
    """Block until the llama.cpp ``/health`` endpoint returns 200."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            conn = HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return
        except (ConnectionError, HTTPException, OSError):
            pass
        time.sleep(1)

    raise TimeoutError(f"Server at {host}:{port} did not become ready within {timeout}s")


def _stop_server(proc: subprocess.Popen | None, serve_args: argparse.Namespace):
    """Shut down a temporary llama.cpp server."""
    if serve_args.container and getattr(serve_args, "name", None):
        try:
            from ramalama.runtime.engine import stop_container

            stop_args = argparse.Namespace(engine=serve_args.engine, ignore=True)
            stop_container(stop_args, serve_args.name)
        except Exception as e:
            logger.debug(f"Failed to stop container {serve_args.name}: {e}")
    elif proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _build_rag_image(args, qdrant_dir: str, image_name: str):
    """Build a minimal OCI image containing the Qdrant database files."""
    containerfile = "FROM scratch\nCOPY . /vector.db\nLABEL ai.ramalama.rag=true\n"

    build_args = argparse.Namespace(
        engine=args.engine,
        dryrun=args.dryrun,
        quiet=getattr(args, "quiet", False),
        image=image_name,
        pull="never",
        network=None,
        oci_runtime=None,
        selinux=False,
        nocapdrop=False,
        device=None,
        podman_keep_groups=False,
        MODEL=None,
        runtime=None,
        port=None,
        subcommand="rag",
    )

    engine = BuildEngine(build_args)
    engine.build_containerfile(containerfile, qdrant_dir, tag=image_name)
