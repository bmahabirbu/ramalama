"""SHA-256 checksum helpers."""

from __future__ import annotations

import hashlib
import os


def generate_sha256_binary(to_hash: bytes, with_sha_prefix: bool = True) -> str:
    """Generates a sha256 for data bytes."""
    h = hashlib.new("sha256")
    h.update(to_hash)
    if with_sha_prefix:
        return f"sha256-{h.hexdigest()}"
    return h.hexdigest()


def generate_sha256(to_hash: str, with_sha_prefix: bool = True) -> str:
    return generate_sha256_binary(to_hash.encode("utf-8"), with_sha_prefix)


def verify_checksum(filename: str) -> bool:
    """Verifies if the SHA-256 checksum of a file matches the checksum in the filename."""
    if not os.path.exists(filename):
        return False

    expected_checksum = ""
    fn_base = os.path.basename(filename)
    if fn_base.startswith("sha256:"):
        expected_checksum = fn_base.split(":")[1]
    elif fn_base.startswith("sha256-"):
        expected_checksum = fn_base.split("-")[1]
    else:
        raise ValueError(f"filename has to start with 'sha256:' or 'sha256-': {fn_base}")

    if len(expected_checksum) != 64:
        raise ValueError("invalid checksum length in filename")

    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest() == expected_checksum
