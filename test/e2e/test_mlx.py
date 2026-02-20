import re
import subprocess
from subprocess import CalledProcessError
from test.conftest import skip_if_apple_silicon, skip_if_no_mlx, skip_if_not_apple_silicon
from test.e2e.utils import RamalamaExecWorkspace, check_output

import pytest

MODEL = "hf://mlx-community/SmolLM-135M-4bit"


@pytest.mark.e2e
def test_runtime_mlx_help_shows_mlx_option():
    """ramalama --runtime=mlx help shows MLX option"""
    result = check_output(["ramalama", "--help"])
    assert "mlx" in result, "MLX should be listed as a runtime option"


@pytest.mark.e2e
@skip_if_not_apple_silicon
@skip_if_no_mlx
def test_runtime_mlx_dryrun_serve_shows_mlx_server_command():
    """ramalama --runtime=mlx --dryrun serve should show the MLX server command with a default-range port"""
    with RamalamaExecWorkspace() as ctx:
        result = ctx.check_output(["ramalama", "--runtime=mlx", "--dryrun", "serve", MODEL])
        assert re.search(r"mlx_lm\.server", result), "should use MLX server command"
        assert re.search(r"--port\s+80[89]\d", result), "should include default-range port"


@pytest.mark.e2e
@skip_if_not_apple_silicon
@skip_if_no_mlx
def test_runtime_mlx_dryrun_serve_with_custom_port():
    """ramalama --runtime=mlx --dryrun serve with a custom port should include the custom port"""
    with RamalamaExecWorkspace() as ctx:
        result = ctx.check_output(["ramalama", "--runtime=mlx", "--dryrun", "serve", "--port", "9090", MODEL])
        assert re.search(r"--port\s+9090", result), "should include custom port"


@pytest.mark.e2e
@skip_if_not_apple_silicon
@skip_if_no_mlx
def test_runtime_mlx_dryrun_serve_with_host():
    """ramalama --runtime=mlx --dryrun serve with a custom host should include the custom host"""
    with RamalamaExecWorkspace() as ctx:
        result = ctx.check_output(["ramalama", "--runtime=mlx", "--dryrun", "serve", "--host", "127.0.0.1", MODEL])
        assert re.search(r"--host\s+127\.0\.0\.1", result), "should include custom host"


@pytest.mark.e2e
@skip_if_apple_silicon
def test_runtime_mlx_serve_fails_on_non_apple_silicon():
    """ramalama --runtime=mlx serve should fail on non-Apple Silicon systems"""
    with RamalamaExecWorkspace() as ctx:
        with pytest.raises(CalledProcessError) as exc_info:
            ctx.check_output(["ramalama", "--runtime=mlx", "serve", MODEL], stderr=subprocess.STDOUT)
        assert exc_info.value.returncode == 22
        assert re.search(
            r"MLX.*Apple Silicon", exc_info.value.output.decode("utf-8")
        ), "should show Apple Silicon requirement error"


@pytest.mark.e2e
@skip_if_not_apple_silicon
@skip_if_no_mlx
def test_runtime_mlx_works_with_ollama_model_format():
    """ramalama --runtime=mlx should work with ollama model format"""
    with RamalamaExecWorkspace() as ctx:
        ollama_model = "ollama://smollm:135m"
        result = ctx.check_output(["ramalama", "--runtime=mlx", "--dryrun", "serve", ollama_model])
        assert re.search(r"mlx_lm\.server", result), "should use MLX server command"


@pytest.mark.e2e
@skip_if_not_apple_silicon
@skip_if_no_mlx
def test_runtime_mlx_works_with_huggingface_model_format():
    """ramalama --runtime=mlx should work with huggingface model format"""
    with RamalamaExecWorkspace() as ctx:
        hf_model = "huggingface://microsoft/DialoGPT-small"
        result = ctx.check_output(["ramalama", "--runtime=mlx", "--dryrun", "serve", hf_model])
        assert re.search(r"mlx_lm\.server", result), "should use MLX server command"
