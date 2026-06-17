"""Importing tract must force transformers to the torch-only backend.

TRACT uses PyTorch via sentence-transformers and never TensorFlow. If TensorFlow
is also installed, transformers auto-imports it during model loading, and a
broken/conflicting TF native library deadlocks on an abseil mutex
(tensorflow pywrap preload_check) on macOS — wedging `import sentence_transformers`.
Setting USE_TF=0 / USE_FLAX=0 at tract import time makes transformers skip TF.
"""
import os
import subprocess
import sys


def test_importing_tract_forces_torch_only_backend() -> None:
    # Clean env so we test tract's own behavior, not an inherited setting.
    env = {k: v for k, v in os.environ.items() if k not in ("USE_TF", "USE_FLAX")}
    out = subprocess.run(
        [sys.executable, "-c",
         "import tract, os; print(os.environ.get('USE_TF'), os.environ.get('USE_FLAX'))"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0 0", out.stdout + out.stderr


def test_explicit_use_tf_is_respected() -> None:
    # An operator who explicitly opts into TF must not be overridden (setdefault).
    env = {**os.environ, "USE_TF": "1"}
    out = subprocess.run(
        [sys.executable, "-c", "import tract, os; print(os.environ['USE_TF'])"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1", out.stdout + out.stderr
