import os
import shutil
import subprocess
from pathlib import Path
from sys import platform
from unittest.mock import MagicMock, Mock, patch

import pytest

from ramalama.cli import configure_subcommands, create_argument_parser, default_image
from ramalama.common import (
    accel_image,
    get_accel,
    populate_volume_from_image,
    rm_until_substring,
    verify_checksum,
)
from ramalama.compat import NamedTemporaryFile
from ramalama.config import DEFAULT_IMAGE, default_config


@pytest.mark.parametrize(
    "input,rm_until,expected",
    [
        ("", "", ""),
        ("huggingface://granite-code", "://", "granite-code"),
        ("hf://granite-code", "://", "granite-code"),
        ("hf.co/granite-code", "hf.co/", "granite-code"),
        (
            "http://huggingface.co/ibm-granite/granite-3b-code-base-2k-GGUF/blob/main/granite-3b-code-base.Q4_K_M.gguf",
            ".co/",
            "ibm-granite/granite-3b-code-base-2k-GGUF/blob/main/granite-3b-code-base.Q4_K_M.gguf",
        ),
        ("modelscope://granite-code", "://", "granite-code"),
        ("ms://granite-code", "://", "granite-code"),
        (
            "file:///tmp/models/granite-3b-code-base.Q4_K_M.gguf",
            "",
            "file:///tmp/models/granite-3b-code-base.Q4_K_M.gguf",
        ),
    ],
)
def test_rm_until_substring(input: str, rm_until: str, expected: str):
    actual = rm_until_substring(input, rm_until)
    assert actual == expected


valid_input = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

tampered_input = """{"model_format":"gguf","model_family":"llama","model_families":["llama"],"model_type":"361.82M","file_type":"Q4_0","architecture":"amd64","os":"linux","rootfs":{"type":"layers","diff_ids":["sha256:f7ae49f9d598730afa2de96fc7dade47f5850446bf813df2e9d739cc8a6c4f29","sha256:62fbfd9ed093d6e5ac83190c86eec5369317919f4b149598d2dbb38900e9faef","sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30","sha256:ca7a9654b5469dc2d638456f31a51a03367987c54135c089165752d9eeb08cd7"]}}

I have been tampered with

"""  # noqa: E501


@pytest.mark.parametrize(
    "input_file_name,content,expected_error,expected_result",
    [
        ("invalidname", "", ValueError, None),
        ("sha256:123", "RamaLama - make working with AI boring through the use of OCI containers.", ValueError, None),
        ("sha256:62fbfd9ed093d6e5ac83190c86eec5369317919f4b149598d2dbb38900e9faef", valid_input, None, True),
        ("sha256-62fbfd9ed093d6e5ac83190c86eec5369317919f4b149598d2dbb38900e9faef", valid_input, None, True),
        ("sha256:16cd1aa2bd52b0e87ff143e8a8a7bb6fcb0163c624396ca58e7f75ec99ef081f", tampered_input, None, False),
    ],
)
def test_verify_checksum(
    input_file_name: str, content: str, expected_error: type[Exception] | None, expected_result: bool
):
    # skip this test case on Windows since colon is not a valid file symbol
    if ":" in input_file_name and platform == "win32":
        return

    full_dir_path = os.path.join(Path(__file__).parent, "verify_checksum")
    file_path = os.path.join(full_dir_path, input_file_name)

    try:
        os.makedirs(full_dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)

        if expected_error is None:
            assert verify_checksum(file_path) == expected_result
            return

        with pytest.raises(expected_error):
            verify_checksum(file_path)
    finally:
        shutil.rmtree(full_dir_path)


@pytest.mark.parametrize(
    "env_override,config_override,expected_result",
    [
        (None, None, f"{DEFAULT_IMAGE}:latest"),
        (f"{DEFAULT_IMAGE}:latest", None, f"{DEFAULT_IMAGE}:latest"),
        (None, f"{DEFAULT_IMAGE}:latest", f"{DEFAULT_IMAGE}:latest"),
        (f"{DEFAULT_IMAGE}:tag", None, f"{DEFAULT_IMAGE}:tag"),
        (None, f"{DEFAULT_IMAGE}:tag", f"{DEFAULT_IMAGE}:tag"),
        (f"{DEFAULT_IMAGE}@sha256:digest", None, f"{DEFAULT_IMAGE}@sha256:digest"),
        (None, f"{DEFAULT_IMAGE}@sha256:digest", f"{DEFAULT_IMAGE}@sha256:digest"),
    ],
)
def test_accel_image(env_override, config_override: str, expected_result: str, monkeypatch):
    monkeypatch.setattr("ramalama.common.get_accel", lambda: "none")
    monkeypatch.setattr("ramalama.common.attempt_to_use_versioned", lambda *args, **kwargs: False)

    with NamedTemporaryFile('w', delete_on_close=False) as f:
        env = {}
        if config_override:
            f.write(f"""\
[ramalama]
image = "{config_override}"
                """)
            f.flush()
            env["RAMALAMA_CONFIG"] = f.name
        else:
            env["RAMALAMA_CONFIG"] = "/dev/null"

        if env_override:
            env["RAMALAMA_IMAGE"] = env_override

        with patch.dict("os.environ", env, clear=True):
            config = default_config()
            with patch("ramalama.cli.get_config", return_value=config):
                default_image.cache_clear()
                parser = create_argument_parser("test_accel_image")
                configure_subcommands(parser)
                assert accel_image(config) == expected_result


@patch("ramalama.common.run_cmd")
@patch("ramalama.common.handle_provider")
def test_apple_vm_returns_result(mock_handle_provider, mock_run_cmd):
    mock_run_cmd.return_value.stdout = b'[{"Name": "myvm"}]'
    mock_handle_provider.return_value = True
    config = object()
    from ramalama.common import apple_vm

    result = apple_vm("podman", config)

    assert result is True
    mock_run_cmd.assert_called_once_with(
        ["podman", "machine", "list", "--format", "json", "--all-providers"], ignore_stderr=True, encoding="utf-8"
    )
    mock_handle_provider.assert_called_once_with({"Name": "myvm"}, config)


class TestGetAccel:
    """Tests for get_accel() which checks /dev/dxg and /dev/dri for vulkan support."""

    def setup_method(self):
        get_accel.cache_clear()

    @patch("ramalama.common.os.path.exists")
    def test_get_accel_returns_vulkan_when_dxg_exists(self, mock_exists):
        mock_exists.side_effect = lambda p: p == "/dev/dxg"
        assert get_accel() == "vulkan"

    @patch("ramalama.common.os.path.exists")
    def test_get_accel_returns_vulkan_when_dri_exists(self, mock_exists):
        # /dev/dxg checked first, returns False; /dev/dri returns True
        mock_exists.side_effect = lambda p: p == "/dev/dri"
        assert get_accel() == "vulkan"

    @patch("ramalama.common.os.path.exists")
    def test_get_accel_returns_none_when_neither_exists(self, mock_exists):
        mock_exists.return_value = False
        assert get_accel() == "none"

    @patch("ramalama.common.os.path.exists")
    def test_get_accel_prefers_dxg_over_dri(self, mock_exists):
        # When both exist, /dev/dxg is checked first and returns vulkan
        mock_exists.side_effect = lambda p: p in ("/dev/dxg", "/dev/dri")
        assert get_accel() == "vulkan"


class TestPopulateVolumeFromImage:
    """Test the populate_volume_from_image function for Docker volume creation"""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with required attributes"""
        model = Mock()
        model.model = "test-registry.io/test-model:latest"
        model.conman = "docker"
        return model

    @patch('subprocess.Popen')
    @patch('ramalama.common.run_cmd')
    def test_populate_volume_success(self, mock_run_cmd, mock_popen, mock_model):
        """Test successful volume population with Docker"""
        output_filename = "model.gguf"

        # Mock the Popen processes for export/tar streaming
        mock_export_proc = MagicMock()
        mock_export_proc.stdout = Mock()
        mock_export_proc.wait.return_value = 0
        mock_export_proc.__enter__ = Mock(return_value=mock_export_proc)
        mock_export_proc.__exit__ = Mock(return_value=None)

        mock_tar_proc = MagicMock()
        mock_tar_proc.wait.return_value = 0
        mock_tar_proc.__enter__ = Mock(return_value=mock_tar_proc)
        mock_tar_proc.__exit__ = Mock(return_value=None)

        mock_popen.side_effect = [mock_export_proc, mock_tar_proc]

        result = populate_volume_from_image(mock_model, Mock(engine="docker"), output_filename)

        assert result.startswith("ramalama-models-")

        assert mock_run_cmd.call_count >= 3
        assert mock_popen.call_count == 2

    @patch('subprocess.Popen')
    @patch('ramalama.common.run_cmd')
    def test_populate_volume_export_failure(self, _, mock_popen, mock_model):
        """Test handling of export process failure"""
        output_filename = "model.gguf"

        # Mock export process failure
        mock_export_proc = MagicMock()
        mock_export_proc.stdout = Mock()
        mock_export_proc.wait.return_value = 1  # Failure
        mock_export_proc.__enter__ = Mock(return_value=mock_export_proc)
        mock_export_proc.__exit__ = Mock(return_value=None)

        mock_tar_proc = MagicMock()
        mock_tar_proc.wait.return_value = 0
        mock_tar_proc.__enter__ = Mock(return_value=mock_tar_proc)
        mock_tar_proc.__exit__ = Mock(return_value=None)

        mock_popen.side_effect = [mock_export_proc, mock_tar_proc]

        with pytest.raises(subprocess.CalledProcessError):
            populate_volume_from_image(mock_model, Mock(engine="docker"), output_filename)

    @patch('subprocess.Popen')
    @patch('ramalama.common.run_cmd')
    def test_populate_volume_tar_failure(self, _, mock_popen, mock_model):
        """Test handling of tar process failure"""
        output_filename = "model.gguf"

        # Mock tar process failure
        mock_export_proc = MagicMock()
        mock_export_proc.stdout = Mock()
        mock_export_proc.wait.return_value = 0
        mock_export_proc.__enter__ = Mock(return_value=mock_export_proc)
        mock_export_proc.__exit__ = Mock(return_value=None)

        mock_tar_proc = MagicMock()
        mock_tar_proc.wait.return_value = 1  # Failure
        mock_tar_proc.__enter__ = Mock(return_value=mock_tar_proc)
        mock_tar_proc.__exit__ = Mock(return_value=None)

        mock_popen.side_effect = [mock_export_proc, mock_tar_proc]

        with pytest.raises(subprocess.CalledProcessError):
            populate_volume_from_image(mock_model, Mock(engine="docker"), output_filename)

    def test_volume_name_generation(self, mock_model):
        """Test that volume names are generated consistently based on model hash"""
        import hashlib

        expected_hash = hashlib.sha256(mock_model.model.encode()).hexdigest()[:12]
        expected_volume = f"ramalama-models-{expected_hash}"

        with patch('subprocess.Popen') as mock_popen, patch('ramalama.common.run_cmd'):
            # Mock successful processes
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.__enter__ = Mock(return_value=mock_proc)
            mock_proc.__exit__ = Mock(return_value=None)
            mock_popen.return_value = mock_proc

            result = populate_volume_from_image(mock_model, Mock(engine="docker"), "test.gguf")
            assert result == expected_volume
