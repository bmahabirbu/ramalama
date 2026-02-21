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

GRANITE_DOCLING_MODEL = "hf://ibm-granite/granite-docling-258M-GGUF"


def run_rag_pipeline(args):
    """Execute the full RAG pipeline.

    1. Pull/verify the Granite Docling GGUF model.
    2. Start a temporary llama.cpp server for document conversion.
    3. Send each document image through the VLM to get structured text.
    4. Chunk the text and embed into a Qdrant vector database.
    5. Package the Qdrant database as a container image.
    """
    from ramalama.rag.chunker import chunk_texts
    from ramalama.rag.converter import GraniteDoclingConverter
    from ramalama.rag.vectordb import store_in_qdrant

    source_path = Path(args.DOCUMENTS)
    image_name = args.IMAGE_NAME
    model_name = getattr(args, "docling_model", GRANITE_DOCLING_MODEL)

    if not source_path.exists():
        raise FileNotFoundError(f"Document path not found: {source_path}")

    if args.engine is None:
        raise ValueError("A container engine (podman or docker) is required to build the RAG image.")

    serve_args = _build_serve_args(args, model_name)
    port = int(serve_args.port)

    perror("Pulling Granite Docling model...")
    model = New(model_name, serve_args)
    model.ensure_model_exists(serve_args)

    set_accel_env_vars()
    cmd = assemble_command_lazy(serve_args)

    perror("Starting Granite Docling inference server...")
    proc = model.serve_nonblocking(serve_args, cmd)

    try:
        _wait_for_server("localhost", port)
        perror("Granite Docling server is ready.")

        converter = GraniteDoclingConverter(api_url=f"http://localhost:{port}")
        texts = converter.convert_directory(source_path)

        if not texts:
            raise ValueError("No documents were successfully converted")

        perror("Chunking and embedding documents...")
        chunks, ids = chunk_texts(texts)
        perror(f"Created {len(chunks)} chunks from {len(texts)} document(s)")

        with tempfile.TemporaryDirectory() as tmpdir:
            qdrant_dir = os.path.join(tmpdir, "qdrant_data")
            store_in_qdrant(chunks, ids, qdrant_dir)

            perror(f"Building container image '{image_name}'...")
            _build_rag_image(args, qdrant_dir, image_name)

        perror(f"RAG image '{image_name}' created successfully.")
    finally:
        _stop_server(proc, serve_args)


def _build_serve_args(args, model_name: str) -> argparse.Namespace:
    """Construct an ``argparse.Namespace`` suitable for an internal llama.cpp serve session."""
    config = get_config()

    serve_args = argparse.Namespace(
        MODEL=model_name,
        subcommand="serve",
        runtime="llama.cpp",
        # Global options carried from the user's invocation
        container=args.container,
        engine=args.engine,
        store=args.store,
        dryrun=args.dryrun,
        debug=args.debug,
        quiet=True,
        noout=True,
        # Container / engine options
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
        # Serve / inference options
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
        # Pull-related options needed by the transport layer
        authfile=None,
        tlsverify=True,
        verify=config.verify,
    )

    serve_args.port = compute_serving_port(serve_args)
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

    raise TimeoutError(f"Granite Docling server at {host}:{port} did not become ready within {timeout}s")


def _stop_server(proc: subprocess.Popen | None, serve_args: argparse.Namespace):
    """Shut down the temporary llama.cpp server."""
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
