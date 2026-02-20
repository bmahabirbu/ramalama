import subprocess

from ramalama.utils.common import MNT_DIR, perror, run_cmd
from ramalama.transports.base import NoRefFileFound, Transport

prefix = "oci://"


class OCI(Transport):
    type = "OCI"

    def __init__(self, model: str, model_store_path: str, conman: str, ignore_stderr: bool = False):
        # Must fix the tag before the base class parses the name
        model_name = model.split('/')[-1]
        if ":" not in model_name:
            model = f"{model}:latest"
        super().__init__(model, model_store_path)

        if not conman:
            raise ValueError("RamaLama OCI Images requires a container engine")
        self.conman = conman
        self.ignore_stderr = ignore_stderr

    def _rm_artifact(self, ignore):
        rm_cmd = [
            self.conman,
            "artifact",
            "rm",
        ]
        if ignore:
            rm_cmd.append("--ignore")
        rm_cmd.append(self.model)

        run_cmd(
            rm_cmd,
            ignore_all=True,
        )

    def pull(self, args):
        if not args.engine:
            raise NotImplementedError("OCI images require a container engine like Podman or Docker")

        conman_args = [args.engine, "pull"]
        if args.quiet:
            conman_args.extend(['--quiet'])
        else:
            # Write message to stderr
            perror(f"Downloading {self.model} ...")
        if str(args.tlsverify).lower() == "false":
            conman_args.extend([f"--tls-verify={args.tlsverify}"])
        if args.authfile:
            conman_args.extend([f"--authfile={args.authfile}"])
        conman_args.extend([self.model])
        run_cmd(conman_args, ignore_stderr=self.ignore_stderr)

    def remove(self, args) -> bool:
        if self.conman is None:
            raise NotImplementedError("OCI Images require a container engine")

        try:
            conman_args = [self.conman, "manifest", "rm", self.model]
            run_cmd(conman_args, ignore_stderr=True)
        except subprocess.CalledProcessError:
            try:
                conman_args = [self.conman, "rmi", f"--force={args.ignore}", self.model]
                run_cmd(conman_args, ignore_stderr=True)
            except subprocess.CalledProcessError:
                try:
                    self._rm_artifact(args.ignore)
                except subprocess.CalledProcessError:
                    raise KeyError(f"Model '{self.model}' not found")
        return True

    def exists(self) -> bool:
        if self.conman is None:
            return False

        conman_args = [self.conman, "image", "inspect", self.model]
        try:
            run_cmd(conman_args, ignore_stderr=True)
            return True
        except Exception:
            conman_args = [self.conman, "artifact", "inspect", self.model]
            try:
                run_cmd(conman_args, ignore_stderr=True)
                return True
            except Exception:
                return False

    def artifact_name(self) -> str:
        conman_args = [
            self.conman,
            "artifact",
            "inspect",
            "--format",
            '{{index  .Manifest.Annotations "org.opencontainers.image.title" }}',
            self.model,
        ]

        return run_cmd(conman_args, ignore_stderr=True).stdout.decode('utf-8').strip()

    def is_artifact(self) -> bool:
        try:
            conman_args = [self.conman, "artifact", "inspect", self.model]
            run_cmd(conman_args, ignore_stderr=True)
            return True
        except (NoRefFileFound, subprocess.CalledProcessError):
            return False

    def mount_cmd(self):
        if self.artifact:
            return f"--mount=type=artifact,src={self.model},destination={MNT_DIR}"
        else:
            return f"--mount=type=image,src={self.model},destination={MNT_DIR},subpath=/models,rw=false"
