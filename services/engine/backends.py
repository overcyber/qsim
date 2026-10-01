"""Backends de computacao do QSim.

O nucleo do simulador e escrito de forma agnostica ao modulo de arrays:
recebe um modulo compativel com a API NumPy (numpy ou cupy) e opera sobre
ele. Isso permite rodar o mesmo codigo em CPU e em qualquer GPU CUDA
(RTX 3090, RTX 4060 Ti e Tesla P40 — a P40, Pascal CC 6.1, nao suporta
cuQuantum, mas roda CuPy normalmente).

Selecao:
  "numpy" | "cpu"  -> NumPy
  "cupy"  | "gpu"  -> CuPy (erro claro se indisponivel)
  "auto"           -> CuPy se instalado e com GPU visivel; senao NumPy
"""
from __future__ import annotations

import numpy as _np

_CUPY = None
_CUPY_ERR: str | None = None


def _try_cupy():
    global _CUPY, _CUPY_ERR
    if _CUPY is not None or _CUPY_ERR is not None:
        return _CUPY
    try:
        import cupy  # type: ignore
        cupy.cuda.runtime.getDeviceCount()
        _CUPY = cupy
    except Exception as err:
        _CUPY_ERR = str(err)
    return _CUPY


def get_backend(name: str = "auto"):
    """Retorna (modulo_xp, nome_efetivo)."""
    key = (name or "auto").lower()
    if key in ("numpy", "cpu"):
        return _np, "numpy"
    if key in ("cupy", "gpu"):
        cp = _try_cupy()
        if cp is None:
            raise RuntimeError(
                f"Backend cupy indisponivel: {_CUPY_ERR or 'cupy nao instalado'}"
            )
        return cp, "cupy"
    if key == "custatevec":
        cp = _try_cupy()
        if cp is None:
            raise RuntimeError(
                f"Backend custatevec requer cupy: {_CUPY_ERR or 'cupy nao instalado'}"
            )
        return cp, "custatevec"
    if key == "auto":
        cp = _try_cupy()
        if cp is not None:
            return cp, "cupy"
        return _np, "numpy"
    raise ValueError(
        f"Backend desconhecido: '{name}' (use numpy, cupy, custatevec ou auto)"
    )


def to_cpu(arr):
    """Traz um array para a memoria da CPU (no-op para NumPy)."""
    if hasattr(arr, "get"):
        return arr.get()
    return _np.asarray(arr)


def available_backends() -> list[str]:
    out = ["numpy"]
    cp = _try_cupy()
    if cp is not None:
        out.append("cupy")
        try:
            import cuquantum  # noqa: F401
            cc = int(cp.cuda.Device().compute_capability)
            if cc >= 70:  # cuStateVec requer Volta+ (P40 Pascal 6.1: apenas cupy)
                out.append("custatevec")
        except Exception:
            pass
    try:
        import numba  # noqa: F401
        out.append("numba")  # acelerador de kernels 1q em CPU, nao um backend proprio
    except Exception:
        pass
    try:
        from services.engine.backend_aer import aer_available, aer_gpu_available
        if aer_available():
            out.append("aer")
            if aer_gpu_available():
                out.append("aer-gpu")
    except Exception:
        pass
    return out
