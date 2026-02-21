"""Name generation and string manipulation helpers."""

from __future__ import annotations

import random
import re
import string

SPLIT_MODEL_PATH_RE = r'(.*?)(?:/)?([^/]*)-00001-of-(\d{5})\.gguf'


def is_split_file_model(model_path):
    """Returns true if ends with -%05d-of-%05d.gguf"""
    return bool(re.match(SPLIT_MODEL_PATH_RE, model_path))


def sanitize_filename(filename: str) -> str:
    return filename.replace(":", "-")


def genname():
    return "ramalama-" + "".join(random.choices(string.ascii_letters + string.digits, k=10))


def rm_until_substring(input: str, substring: str) -> str:
    pos = input.find(substring)
    if pos == -1:
        return input
    return input[pos + len(substring) :]
