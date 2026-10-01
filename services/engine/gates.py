"""Definicao das portas quanticas (matrizes unitarias) do QSim.

Todas as matrizes usam complex128. Portas parametrizadas sao funcoes
que recebem angulos em radianos e retornam a matriz correspondente.
"""
from __future__ import annotations

import numpy as np

SQRT2_INV = 1.0 / np.sqrt(2.0)

# ---------------------------------------------------------------------------
# Portas fixas de 1 qubit
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = SQRT2_INV * np.array([[1, 1], [1, -1]], dtype=np.complex128)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Portas parametrizadas de 1 qubit
# ---------------------------------------------------------------------------
def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    return np.array([[c, s], [s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def p(lam: float) -> np.ndarray:
    """Porta de fase P(lambda)."""
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)


def u(theta: float, phi: float, lam: float) -> np.ndarray:
    """Porta universal U(theta, phi, lambda) - convencao OpenQASM 3."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# Portas de 2 qubits (ordem de qubits: [controle, alvo] -> big-endian local)
# ---------------------------------------------------------------------------
CX = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=np.complex128,
)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
SWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=np.complex128,
)
ISWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1j, 0],
     [0, 1j, 0, 0],
     [0, 0, 0, 1]],
    dtype=np.complex128,
)


def cp(lam: float) -> np.ndarray:
    return np.diag([1, 1, 1, np.exp(1j * lam)]).astype(np.complex128)


def rxx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    m = np.eye(4, dtype=np.complex128) * c
    m[0, 3] = m[1, 2] = m[2, 1] = m[3, 0] = s
    return m


def rzz(theta: float) -> np.ndarray:
    e_m = np.exp(-1j * theta / 2)
    e_p = np.exp(1j * theta / 2)
    return np.diag([e_m, e_p, e_p, e_m]).astype(np.complex128)


# ---------------------------------------------------------------------------
# Portas de 3 qubits
# ---------------------------------------------------------------------------
CCX = np.eye(8, dtype=np.complex128)
CCX[6, 6], CCX[7, 7], CCX[6, 7], CCX[7, 6] = 0, 0, 1, 1

CSWAP = np.eye(8, dtype=np.complex128)
CSWAP[5, 5], CSWAP[6, 6], CSWAP[5, 6], CSWAP[6, 5] = 0, 0, 1, 1


# ---------------------------------------------------------------------------
# Registro: nome -> (matriz | fabrica, n_qubits, n_params)
# ---------------------------------------------------------------------------
FIXED = {
    "i": (I, 1), "id": (I, 1), "x": (X, 1), "y": (Y, 1), "z": (Z, 1),
    "h": (H, 1), "s": (S, 1), "sdg": (SDG, 1), "t": (T, 1), "tdg": (TDG, 1),
    "sx": (SX, 1),
    "cx": (CX, 2), "cnot": (CX, 2), "cz": (CZ, 2), "swap": (SWAP, 2),
    "iswap": (ISWAP, 2),
    "ccx": (CCX, 3), "toffoli": (CCX, 3), "cswap": (CSWAP, 3),
    "fredkin": (CSWAP, 3),
}

PARAMETRIC = {
    "rx": (rx, 1, 1), "ry": (ry, 1, 1), "rz": (rz, 1, 1), "p": (p, 1, 1),
    "phase": (p, 1, 1), "u": (u, 1, 3),
    "cp": (cp, 2, 1), "rxx": (rxx, 2, 1), "rzz": (rzz, 2, 1),
}


def resolve(name: str, params: list[float] | None) -> tuple[np.ndarray, int]:
    """Retorna (matriz, n_qubits) para a porta pedida.

    Levanta ValueError para porta desconhecida ou numero errado de parametros.
    """
    key = name.lower()
    if key in FIXED:
        if params:
            raise ValueError(f"Porta '{name}' nao aceita parametros")
        mat, nq = FIXED[key]
        return mat, nq
    if key in PARAMETRIC:
        factory, nq, nparams = PARAMETRIC[key]
        params = params or []
        if len(params) != nparams:
            raise ValueError(
                f"Porta '{name}' requer {nparams} parametro(s), recebeu {len(params)}"
            )
        return factory(*params), nq
    raise ValueError(f"Porta desconhecida: '{name}'")
