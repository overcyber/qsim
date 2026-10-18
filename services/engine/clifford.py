"""Simulador Clifford por geradores de estabilizadores (P2.1 do plano).

Circuitos restritos a {H, X, Y, Z, S, SDG, CX, CZ, SWAP, I} sao simulados
manipulando os n geradores de Pauli do estado (mascaras X/Z em uint64
empacotados + fase em Z4) em vez do vetor de estado: O(n^2 bits) de memoria
e O(n) operacoes de mascara por porta — GHZ de 1000 qubits e trivial.

Formulacao (sem destabilizadores, mais facil de auditar que o CHP classico):

  - Portas: conjugacao dos geradores (G -> U G Udaga) por operacoes de
    coluna identicas as do CHP.
  - Medida de Z_q com algum gerador tendo X/Y em q (resultado aleatorio):
      1. elimina X_q dos outros geradores usando o pivo (produtos de
         linhas com fase por quarter-turns: +1 nos pares ciclicos X.Y,
         Y.Z, Z.X; -1 nos anticciclicos);
      2. pivo <- (-1)^m Z_q com m aleatorio;
      3. elimina Z_q dos outros geradores usando o novo pivo (e isto que
         propaga a correlacao — ex.: no Bell, Z0Z1 vira ±Z1).
  - Medida deterministica (nenhum gerador tem X/Y em q): elimina a coluna
    z_q ate sobrar um unico pivo; se o pivo for exatamente ±Z_q, o
    resultado e a fase dele; senao resolve a combinacao por eliminacao
    de Gauss em GF(2) (caminho geral, O(n^3/64)).

Criterio de aceite (tests/test_clifford.py): crosscheck de distribuicoes
contra o statevector em circuitos aleatorios + casos deterministicos.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

CLIFFORD_GATES = {"h", "x", "y", "z", "s", "sdg", "i", "id",
                  "cx", "cnot", "cz", "swap"}

_WORD = 64


def _build_cx_phase() -> np.ndarray:
    """Tabela de fase do CNOT por combinacao (x_a, z_a, x_b, z_b) da linha,
    derivada no import por conjugacao matricial exata — sem depender de
    regra decorada. Retorna array[16] com 0/1 (fase += 2*valor)."""
    x_m = np.array([[0, 1], [1, 0]], dtype=complex)
    z_m = np.array([[1, 0], [0, -1]], dtype=complex)
    y_m = 1j * x_m @ z_m
    i_m = np.eye(2, dtype=complex)
    can = {(0, 0): i_m, (1, 0): x_m, (0, 1): z_m, (1, 1): y_m}
    u = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)      # ctrl=a (MSB), tgt=b
    lut = np.zeros(16, dtype=np.uint8)
    for (xa, za), (xb, zb) in itertools.product(can, repeat=2):
        g = np.kron(can[(xa, za)], can[(xb, zb)])
        conj = u @ g @ u.T
        base = np.kron(can[(xa, za ^ zb)], can[(xb ^ xa, zb)])
        idx = int(np.argmax(np.abs(base)))
        coeff = conj.flat[idx] / base.flat[idx]
        k = int(round(2 * np.angle(coeff) / np.pi)) % 4
        assert k in (0, 2), (xa, za, xb, zb, k)
        lut[xa * 8 + za * 4 + xb * 2 + zb] = k // 2
    return lut


_CX_PHASE = _build_cx_phase()


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int64)
    out = np.zeros(x.shape, dtype=np.int64)
    v = x.astype(np.uint64).copy()
    while np.any(v):
        out += (v & 1).astype(np.int64)
        v >>= np.uint64(1)
    return out


class Tableau:
    """Estado de n qubits pelos seus n geradores de estabilizador."""

    def __init__(self, n: int) -> None:
        self.n = n
        w = (n + _WORD - 1) // _WORD
        self.xmask = np.zeros((n, w), dtype=np.uint64)   # mascaras X
        self.zmask = np.zeros((n, w), dtype=np.uint64)   # mascaras Z
        self.p = np.zeros(n, dtype=np.uint8)         # fase i^p
        for q in range(n):
            self.zmask[q, q // _WORD] |= np.uint64(1 << (q % _WORD))  # S_q = Z_q

    # ---------------- helpers de coluna ----------------
    def _col(self, arr: np.ndarray, q: int) -> np.ndarray:
        return ((arr[:, q // _WORD] >> np.uint64(q % _WORD)) & 1).astype(np.int8)

    def _flip_col(self, arr: np.ndarray, q: int, rows: np.ndarray) -> None:
        arr[rows, q // _WORD] ^= np.uint64(1 << (q % _WORD))

    # ---------------- produto de linhas com fase ----------------
    def _product_phase(self, h: int, i: int) -> int:
        xh, zh, xi, zi = self.xmask[h], self.zmask[h], self.xmask[i], self.zmask[i]
        plus = (_popcount_u64(xh & ~zh & (xi & zi)).sum()
                + _popcount_u64(xh & zh & (zi & ~xi)).sum()
                + _popcount_u64(~xh & zh & (xi & ~zi)).sum())
        minus = (_popcount_u64(xh & ~zh & (zi & ~xi)).sum()
                 + _popcount_u64(xh & zh & (xi & ~zi)).sum()
                 + _popcount_u64(~xh & zh & (xi & zi)).sum())
        return int(plus - minus)

    def _row_times(self, h: int, i: int) -> None:
        self.p[h] = (self.p[h] + self.p[i] + self._product_phase(h, i)) % 4
        self.xmask[h] ^= self.xmask[i]
        self.zmask[h] ^= self.zmask[i]

    # ---------------- portas (conjugacao G -> U G Udaga) ----------------
    def h(self, q: int) -> None:
        xq = self._col(self.xmask, q).astype(bool)
        zq = self._col(self.zmask, q).astype(bool)
        self.p = (self.p + 2 * (xq & zq).astype(np.uint8)) % 4
        diff = np.nonzero(xq ^ zq)[0]
        self._flip_col(self.xmask, q, diff)
        self._flip_col(self.zmask, q, diff)

    def s(self, q: int) -> None:
        xq = self._col(self.xmask, q).astype(bool)
        zq = self._col(self.zmask, q).astype(bool)
        only_x = np.nonzero(xq & ~zq)[0]           # X -> Y
        x_and_z = np.nonzero(xq & zq)[0]           # Y -> -X
        self._flip_col(self.zmask, q, only_x)
        self._flip_col(self.zmask, q, x_and_z)
        self.p[x_and_z] = (self.p[x_and_z] + 2) % 4

    def sdg(self, q: int) -> None:
        xq = self._col(self.xmask, q).astype(bool)
        zq = self._col(self.zmask, q).astype(bool)
        only_x = np.nonzero(xq & ~zq)[0]           # X -> -Y
        x_and_z = np.nonzero(xq & zq)[0]           # Y -> X
        self._flip_col(self.zmask, q, only_x)
        self._flip_col(self.zmask, q, x_and_z)
        self.p[only_x] = (self.p[only_x] + 2) % 4

    def x(self, q: int) -> None:
        self.p = (self.p + 2 * self._col(self.zmask, q)) % 4

    def y(self, q: int) -> None:
        # Y e hermitiana: YXYdag = -X, YZYdag = -Z, YYYdag = Y.
        # Mascaras inalteradas; fase += 2 quando o fator no qubit e X ou Z.
        xq = self._col(self.xmask, q)
        zq = self._col(self.zmask, q)
        self.p = (self.p + 2 * (xq ^ zq).astype(np.uint8)) % 4

    def z(self, q: int) -> None:
        self.p = (self.p + 2 * self._col(self.xmask, q)) % 4

    def cx(self, a: int, b: int) -> None:
        wa, wb = a // _WORD, b // _WORD
        sa, sb = np.uint64(1 << (a % _WORD)), np.uint64(1 << (b % _WORD))
        xa = (self.xmask[:, wa] & sa) != 0
        za = (self.zmask[:, wa] & sa) != 0
        xb = (self.xmask[:, wb] & sb) != 0
        zb = (self.zmask[:, wb] & sb) != 0          # z_b ANTES de mudar
        idx = (xa.astype(np.uint8) * 8 + za.astype(np.uint8) * 4
               + xb.astype(np.uint8) * 2 + zb.astype(np.uint8))
        self.p = (self.p + 2 * _CX_PHASE[idx]) % 4
        self.xmask[:, wb] = np.where(xa, self.xmask[:, wb] ^ sb, self.xmask[:, wb])
        self.zmask[:, wa] = np.where(zb, self.zmask[:, wa] ^ sa, self.zmask[:, wa])

    def cz(self, a: int, b: int) -> None:
        self.h(b)
        self.cx(a, b)
        self.h(b)

    def swap(self, a: int, b: int) -> None:
        wa, wb = a // _WORD, b // _WORD
        sa, sb = np.uint64(1 << (a % _WORD)), np.uint64(1 << (b % _WORD))
        for arr in (self.xmask, self.zmask):
            if wa == wb:
                word = arr[:, wa]
                ba = (word & sa) != 0
                bb = (word & sb) != 0
                arr[:, wa] = ((word & ~sa & ~sb)
                              | np.where(bb, sa, np.uint64(0))
                              | np.where(ba, sb, np.uint64(0)))
            else:
                ba = (arr[:, wa] & sa) != 0
                bb = (arr[:, wb] & sb) != 0
                newa = (arr[:, wa] & ~sa) | np.where(bb, sa, np.uint64(0))
                newb = (arr[:, wb] & ~sb) | np.where(ba, sb, np.uint64(0))
                arr[:, wa] = newa
                arr[:, wb] = newb

    # ---------------- medida de Z_q ----------------
    def measure(self, q: int, rng) -> int:
        n = self.n
        xcol = self._col(self.xmask, q)
        if xcol.any():
            # ---- aleatorio: colapsa e propaga a correlacao ----
            pivots = np.nonzero(xcol)[0]
            p = int(pivots[0])
            for i in pivots[1:]:
                self._row_times(int(i), p)         # X_q so resta no pivo
            m = int(rng.integers(0, 2))
            self.xmask[p] = 0
            self.zmask[p] = 0
            self.zmask[p, q // _WORD] |= np.uint64(1 << (q % _WORD))
            self.p[p] = 2 * m
            zrows = np.nonzero(self._col(self.zmask, q))[0]
            for i in zrows:
                if i != p:
                    self._row_times(int(i), p)     # Z_q so resta no pivo
            return m

        # ---- deterministico ----
        zrows = np.nonzero(self._col(self.zmask, q))[0]
        if zrows.size == 0:                          # (nao usar .any():
            return 0                                 # indice 0 e falsy!)
        p = int(zrows[0])
        for i in zrows[1:]:
            self._row_times(int(i), p)             # z_q so resta no pivo
        if not self.xmask[p].any() and int(_popcount_u64(self.zmask[p]).sum()) == 1:
            return int(self.p[p] == 2)             # pivo e exatamente +-Z_q
        return self._deterministic_by_solve(q)

    def _deterministic_by_solve(self, q: int) -> int:
        """Caminho geral: RREF completo das linhas (produtos de grupo com
        fase rastreada). Apos a reducao, se Z_q pertence ao grupo, ele e a
        unica linha com bit em z_q — o resultado e a fase dela."""
        x, z, p = self.xmask.copy(), self.zmask.copy(), self.p.copy()
        n = len(p)

        def phase_of(h: int, i: int) -> int:
            xh, zh, xi, zi = x[h], z[h], x[i], z[i]
            plus = (_popcount_u64(xh & ~zh & (xi & zi)).sum()
                    + _popcount_u64(xh & zh & (zi & ~xi)).sum()
                    + _popcount_u64(~xh & zh & (xi & ~zi)).sum())
            minus = (_popcount_u64(xh & ~zh & (zi & ~xi)).sum()
                     + _popcount_u64(xh & zh & (xi & ~zi)).sum()
                     + _popcount_u64(~xh & zh & (xi & zi)).sum())
            return int(plus - minus)

        def bit(src: np.ndarray, row: int, c: int) -> int:
            return (int(src[row, c // _WORD]) >> (c % _WORD)) & 1

        remaining = set(range(n))
        for col in range(2 * n):                      # colunas x_0..x_{n-1},
            src = x if col < n else z                 # depois z_0..z_{n-1}
            c = col if col < n else col - n
            piv = next((i for i in remaining if bit(src, i, c)), None)
            if piv is None:
                continue
            for i in range(n):                        # RREF completo: elimina
                if i != piv and bit(src, i, c):       # tambem das linhas-pivo
                    p[i] = (p[i] + p[piv] + phase_of(i, piv)) % 4
                    x[i] ^= x[piv]
                    z[i] ^= z[piv]
            remaining.discard(piv)                    # anteriores
        w = q // _WORD
        for i in range(n):
            if ((int(z[i, w]) >> (q % _WORD)) & 1) \
                    and int(_popcount_u64(z[i]).sum()) == 1 and not x[i].any():
                return int(p[i] == 2)
        return 0

    def reset(self, q: int, rng) -> None:
        if self.measure(q, rng) == 1:
            self.x(q)


# ---------------------------------------------------------------------------
# Execucao de circuitos
# ---------------------------------------------------------------------------
def is_clifford_circuit(ops) -> bool:
    for op in ops:
        if not isinstance(op, dict):
            return False
        kind = op.get("op", "gate")
        if kind in ("barrier", "measure", "reset"):
            continue
        if kind != "gate":
            return False
        gate = str(op.get("gate", "")).lower()
        if gate not in CLIFFORD_GATES or op.get("params"):
            return False
    return True


def _apply_gate(tab: Tableau, gate: str, targets: list[int]) -> None:
    if gate == "h":
        tab.h(targets[0])
    elif gate == "x":
        tab.x(targets[0])
    elif gate == "y":
        tab.y(targets[0])
    elif gate == "z":
        tab.z(targets[0])
    elif gate == "s":
        tab.s(targets[0])
    elif gate == "sdg":
        tab.sdg(targets[0])
    elif gate in ("i", "id"):
        pass
    elif gate in ("cx", "cnot"):
        tab.cx(targets[0], targets[1])
    elif gate == "cz":
        tab.cz(targets[0], targets[1])
    elif gate == "swap":
        tab.swap(targets[0], targets[1])
    else:
        raise ValueError(f"Porta '{gate}' nao e Clifford")


def run_clifford(circuit: dict, should_cancel=None, max_qubits: int = 2048):
    """Executa um circuito Clifford por geradores. Retorna SimResult (counts).

    Suporta medicao/reset intermediarios e shots com seed. Nao suporta
    ruido, observaveis, Bloch ou statevector (erro claro se pedidos).
    """
    from .simulator import JobCancelled, SimResult  # import tardio: sem ciclo

    t0 = time.perf_counter()
    cancel = should_cancel or (lambda: False)
    n = int(circuit["num_qubits"])
    shots = int(circuit.get("shots", 1024))
    seed = circuit.get("seed")
    ops = circuit["operations"]

    if not 1 <= n <= max_qubits:
        raise ValueError(
            f"backend clifford: num_qubits deve estar entre 1 e {max_qubits}"
        )
    for flag in ("return_statevector", "return_bloch", "return_entanglement"):
        if circuit.get(flag):
            raise ValueError(
                f"backend clifford nao suporta {flag} (use o statevector)"
            )
    if circuit.get("observables"):
        raise ValueError("backend clifford nao suporta observaveis")
    noise = circuit.get("noise") or {}
    if any(noise.get(k) for k in ("depolarizing", "amplitude_damping",
                                  "readout_error")) or \
            noise.get("per_gate") or noise.get("per_qubit"):
        raise ValueError("backend clifford nao suporta ruido")

    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    for _ in range(shots):
        if cancel():
            raise JobCancelled()
        tab = Tableau(n)
        classical: dict[int, int] = {}
        for op in ops:
            kind = op.get("op", "gate")
            if kind == "barrier":
                continue
            if kind == "measure":
                for q in op["targets"]:
                    classical[int(q)] = tab.measure(int(q), rng)
                continue
            if kind == "reset":
                for q in op["targets"]:
                    tab.reset(int(q), rng)
                continue
            _apply_gate(tab, str(op["gate"]).lower(),
                        [int(q) for q in op["targets"]])
        bits = 0
        for q in range(n):
            b = classical.get(q)
            if b is None:
                b = tab.measure(q, rng)
            bits |= b << q
        key = format(bits, f"0{n}b")
        counts[key] = counts.get(key, 0) + 1

    return SimResult(
        counts=dict(sorted(counts.items())),
        shots=shots,
        num_qubits=n,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metadata={"mode": "clifford", "engine": "tableau",
                  "shots": shots, "original_ops": len(ops)},
    )
