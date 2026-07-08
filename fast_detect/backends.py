"""Array-module abstraction so the batched kernels run unchanged on NumPy
(astro3 + Mac CPU, BLAS-threaded) or MLX (Apple-Metal GPU).

The kernels only need: elementwise math (sin/cos/sqrt/exp/log), matmul/tensordot,
reductions (sum/max/argmax along an axis), broadcasting, boolean ops, and array
construction from host NumPy. We deliberately AVOID backend linear-algebra
(``inv``/``solve``/``lstsq``) in the hot path — any small (3x3) system is
pre-inverted once on the host in float64 and uploaded — so the abstraction stays
thin and MLX's limited linalg is never required.

Usage:
    xp = get_xp("numpy")        # or "mlx"
    a = xp.asarray(np_array)    # host numpy -> backend array
    ...                          # operate with xp
    out = to_numpy(result)      # backend array -> host numpy
"""

import numpy as np


def mlx_available():
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend(name):
    """Map a requested backend name to an available one, with graceful fallback.

    'auto' -> mlx if importable else numpy. An explicit 'mlx' that is not
    importable raises (so a misconfigured run fails loudly rather than silently
    running on the CPU).
    """
    name = (name or "numpy").lower()
    if name == "auto":
        return "mlx" if mlx_available() else "numpy"
    if name == "mlx" and not mlx_available():
        raise RuntimeError(
            "backend='mlx' requested but mlx is not importable; "
            "`pip install mlx` (Apple Silicon only) or use backend='numpy'.")
    if name not in ("numpy", "mlx"):
        raise ValueError("unknown backend %r (use numpy|mlx|auto)" % name)
    return name


class _NumpyBackend:
    """Thin wrapper exposing the subset of the API the kernels use."""
    name = "numpy"
    float_dtype = np.float64
    index_dtype = np.int64  # dtype for on-backend running-argmax index buffers

    def __init__(self):
        self._np = np
        # passthroughs
        for fn in ("sin", "cos", "sqrt", "exp", "log", "abs", "where",
                   "maximum", "minimum", "clip", "arctan2", "stack",
                   "concatenate", "zeros", "ones", "full", "arange",
                   "matmul", "einsum", "tensordot", "isfinite"):
            setattr(self, fn, getattr(np, fn))

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype or self.float_dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def astype(self, x, dtype):
        return np.asarray(x).astype(dtype)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def max(self, x, axis=None):
        return np.max(x, axis=axis)

    def argmax(self, x, axis=None):
        return np.argmax(x, axis=axis)

    def take_along_axis(self, x, idx, axis):
        return np.take_along_axis(x, idx, axis=axis)


class _MlxBackend:
    """MLX (Metal) backend. Defaults to float32 on GPU.

    float64 policy (probed on mlx 0.31.2, M-series):
      **float64 is NOT supported on the GPU** — ``mx.array(x, dtype=mx.float64)``
      raises ``ValueError('float64 is not supported on the GPU')``. So any
      float64-precision fidelity work (the 2a peak-flip fix) CANNOT be an
      on-device float64 argmax; it must fall back to a **host** float64
      re-evaluation of the top-K candidate frequencies (or a deterministic
      tie-break). The hot-path GEMM + running-argmax reduction stay float32
      on-device. Verified ops: argmax(axis) [returns uint32], max(axis), where,
      maximum, minimum, take_along_axis, arange — the running-max ``where`` idiom
      matches numpy argmax exactly.
    """
    name = "mlx"

    def __init__(self):
        import mlx.core as mx
        self.mx = mx
        self.float_dtype = mx.float32
        self.index_dtype = mx.uint32  # mx.argmax returns uint32
        for fn in ("sin", "cos", "sqrt", "exp", "log", "abs", "where",
                   "maximum", "minimum", "clip", "arctan2", "stack",
                   "concatenate", "zeros", "ones", "full", "arange",
                   "matmul", "einsum", "tensordot", "isfinite"):
            if hasattr(mx, fn):
                setattr(self, fn, getattr(mx, fn))

    def asarray(self, x, dtype=None):
        return self.mx.array(np.asarray(x, dtype=np.float32)
                             if dtype is None else np.asarray(x, dtype=dtype))

    def to_numpy(self, x):
        return np.array(x)

    def astype(self, x, dtype):
        return x.astype(dtype)

    def sum(self, x, axis=None):
        return self.mx.sum(x, axis=axis)

    def max(self, x, axis=None):
        return self.mx.max(x, axis=axis)

    def argmax(self, x, axis=None):
        return self.mx.argmax(x, axis=axis)

    def take_along_axis(self, x, idx, axis):
        return self.mx.take_along_axis(x, idx, axis=axis)


_CACHE = {}


def get_xp(name="numpy"):
    """Return a backend object (cached) for the resolved backend name."""
    resolved = resolve_backend(name)
    if resolved not in _CACHE:
        _CACHE[resolved] = _NumpyBackend() if resolved == "numpy" else _MlxBackend()
    return _CACHE[resolved]


def to_numpy(x):
    return np.array(x)
