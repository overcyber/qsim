"""Kernels Numba para aceleracao de portas de 1 qubit em CPU (EXPERIMENTAL).

O caminho generico do simulador usa moveaxis + matmul do NumPy, que aloca
um estado intermediario. Estes kernels aplicam a porta in-place, em paralelo
sobre os nucleos da CPU (prange), sem alocacao — ganho relevante a partir de
~18 qubits em maquinas com muitos nucleos (dual Xeon 28c).

Ativacao: QSIM_USE_NUMBA=1 no ambiente do worker (requer pip install numba).
O primeiro uso de cada dtype compila o kernel (alguns segundos, cacheado em
disco); execucoes seguintes sao imediatas.

Validacao: tests/test_gpu.py compara numba vs caminho NumPy puro.

Convencao de indices: little-endian — o bit de posicao t do indice do vetor
de estado corresponde ao qubit t.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

    prange = range  # type: ignore


@njit(parallel=True, fastmath=False, cache=True)
def apply_1q(state, m00, m01, m10, m11, target, n):  # pragma: no cover
    """Aplica a matriz 2x2 [[m00,m01],[m10,m11]] ao qubit `target`,
    in-place, paralelizando sobre os pares de amplitudes."""
    half = 1 << (n - 1)
    mask_low = (1 << target) - 1
    bit = 1 << target
    for k in prange(half):
        low = k & mask_low
        high = (k >> target) << (target + 1)
        i0 = high | low
        i1 = i0 | bit
        a = state[i0]
        b = state[i1]
        state[i0] = m00 * a + m01 * b
        state[i1] = m10 * a + m11 * b


@njit(parallel=True, fastmath=False, cache=True)
def apply_1q_ctrl(state, m00, m01, m10, m11, target, control, n):  # pragma: no cover
    """Versao controlada: aplica a matriz 2x2 ao qubit `target` apenas nos
    ramos em que o qubit `control` vale 1."""
    half = 1 << (n - 1)
    mask_low = (1 << target) - 1
    bit = 1 << target
    cbit = 1 << control
    for k in prange(half):
        low = k & mask_low
        high = (k >> target) << (target + 1)
        i0 = high | low
        if i0 & cbit:
            i1 = i0 | bit
            a = state[i0]
            b = state[i1]
            state[i0] = m00 * a + m01 * b
            state[i1] = m10 * a + m11 * b


def warmup(dtype=np.complex128) -> None:
    """Forca a compilacao JIT fora do caminho critico (chamar no boot do
    worker quando QSIM_USE_NUMBA=1)."""
    if not HAS_NUMBA:
        return
    state = np.zeros(4, dtype=dtype)
    state[0] = 1.0
    apply_1q(state, 0.0 + 0j, 1.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0, 2)
    apply_1q_ctrl(state, 0.0 + 0j, 1.0 + 0j, 1.0 + 0j, 0.0 + 0j, 1, 0, 2)
