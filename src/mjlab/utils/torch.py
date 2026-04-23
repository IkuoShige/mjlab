import os

import torch
from packaging.version import parse


def configure_torch_backends(allow_tf32: bool = True, deterministic: bool = False):
  """Configure PyTorch CUDA and cuDNN backends for performance/reproducibility.

  Args:
    allow_tf32: If True, use TF32 precision for faster computation on Ampere+ GPUs. If
      False, use standard IEEE FP32 precision.
    deterministic: If True, use deterministic algorithms (slower but reproducible).
      If False, allow cuDNN to benchmark and select fastest algorithms.

  Note:
    TF32 uses reduced precision (10-bit mantissa vs 23-bit for FP32) for internal
    matrix multiplications providing a speedup with minimal impact on accuracy.

    See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere for details.
  """
  torch_version = parse(torch.__version__.split("+")[0])  # Handle e.g., "2.9.0+cu118".
  if torch_version >= parse("2.9.0"):
    _configure_29(allow_tf32)
  else:
    _configure_pre29(allow_tf32)

  # Opt-out: some environments have a broken cuDNN install where
  # torch._cudnn_rnn_flatten_weight / conv1d raise CUDNN_STATUS_NOT_INITIALIZED
  # even on a trivial tensor. Set MJLAB_DISABLE_CUDNN=1 to fall back to native
  # kernels (slower for LSTM, fine for most else).
  if os.environ.get("MJLAB_DISABLE_CUDNN") == "1":
    torch.backends.cudnn.enabled = False
  else:
    torch.backends.cudnn.benchmark = not deterministic  # Find fastest algorithms.
    torch.backends.cudnn.deterministic = deterministic  # Reproducibility.


def _configure_29(allow_tf32: bool):
  """Configure PyTorch CUDA and cuDNN backends for PyTorch 2.9+."""
  # tf32 for performance, ieee for full FP32 accuracy.
  precision = "tf32" if allow_tf32 else "ieee"
  torch.backends.cuda.matmul.fp32_precision = precision
  torch.backends.cudnn.fp32_precision = precision  # type: ignore


def _configure_pre29(allow_tf32: bool):
  """Configure PyTorch CUDA and cuDNN backends for PyTorch <2.9."""
  torch.backends.cuda.matmul.allow_tf32 = allow_tf32
  torch.backends.cudnn.allow_tf32 = allow_tf32
