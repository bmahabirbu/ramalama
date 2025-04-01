import os
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse

from ramalama.common import accel_image, download_file, get_accel_env_vars, run_cmd, set_accel_env_vars
from ramalama.config import CONFIG


class Rag:
    model = ""
    target = ""

    def __init__(self, target):
        self.target = target
        set_accel_env_vars()

    def build(self, source, target, args):
        print(f"\nBuilding {target}...")
        contextdir = os.path.dirname(source)
        src = os.path.basename(source)
        print(f"adding {src}...")
        cfile = f"""\
FROM scratch
COPY {src} /vector.db
"""
        containerfile = tempfile.NamedTemporaryFile(dir=source)
        # Open the file for writing.
        with open(containerfile.name, 'w') as c:
            c.write(cfile)
        if args.debug:
            print(f"\nContainerfile: {containerfile.name}\n{cfile}")
        imageid = (
            run_cmd(
                [
                    args.engine,
                    "build",
                    "--no-cache",
                    "--network=none",
                    "-q",
                    "-t",
                    target,
                    "-f",
                    containerfile.name,
                    contextdir,
                ],
                debug=args.debug,
            )
            .stdout.decode("utf-8")
            .strip()
        )
        return imageid

    def _handle_docs_path(self, path, docs_path, exec_args):
        """Adds a volume mount if path exists, otherwise downloads from URL."""
        if os.path.exists(path):
            fpath = os.path.realpath(path)
            exec_args += ["-v", f"{fpath}:/docs/{fpath}:ro,z"]
            return False
        try:
            parsed = urlparse(path)
            if parsed.scheme == "" or parsed.path == "":
                raise ValueError(f"{path} does not exist")
            dpath = docs_path + parsed.path
            os.makedirs(os.path.dirname(dpath), exist_ok=True)
            download_file(path, dpath)
            return True  # docsdb was used
        except RuntimeError as e:
            shutil.rmtree(docs_path, ignore_errors=True)
            raise e

    def generate(self, args):
        # force accel_image to use -rag version
        args.rag = "rag"
        args.image = accel_image(CONFIG, args)

        if not args.container:
            raise KeyError("rag command requires a container. Can not be run with --nocontainer option.")
        if not args.engine or args.engine == "":
            raise KeyError("rag command requires a container. Can not be run without a container engine.")

        tmpdir = "."
        if not os.access(tmpdir, os.W_OK):
            tmpdir = "/tmp"

        args.image = accel_image(CONFIG, args)
        docsdb = tempfile.TemporaryDirectory(dir=tmpdir, prefix='RamaLama_docs_')
        docsdb_used = False
        exec_args = [args.engine, "run", "--rm"]
        if args.network:
            exec_args += ["--network", args.network]

        for path in args.PATH:
            if self._handle_docs_path(path, docsdb.name, exec_args):
                docsdb_used = True

        if docsdb_used:
            exec_args += ["-v", f"{docsdb.name}:/docs/{docsdb.name}:ro,Z"]

        ragdb = tempfile.TemporaryDirectory(dir=tmpdir, prefix='RamaLama_rag_')
        dbdir = os.path.join(ragdb.name, "vectordb")
        os.mkdir(dbdir)
        exec_args += ["-v", f"{dbdir}:/output:z"]
        for k, v in get_accel_env_vars().items():
            # Special case for Cuda
            if k == "CUDA_VISIBLE_DEVICES":
                if os.path.basename(args.engine) == "docker":
                    exec_args += ["--gpus", "all"]
                else:
                    # newer Podman versions support --gpus=all, but < 5.0 do not
                    exec_args += ["--device", "nvidia.com/gpu=all"]

            exec_args += ["-e", f"{k}={v}"]

        exec_args += [args.image]
        exec_args += ["doc2rag", "/output", "/docs/"]
        try:
            run_cmd(exec_args, debug=args.debug)
            print(self.build(dbdir, self.target, args))
        except subprocess.CalledProcessError as e:
            raise e
        finally:
            shutil.rmtree(ragdb.name, ignore_errors=True)
            shutil.rmtree(docsdb.name, ignore_errors=True)
