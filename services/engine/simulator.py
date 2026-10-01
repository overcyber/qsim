"""QSim - Simulador de computador quantico por vetor de estado.

Nucleo teorico-empirico com recursos de pesquisa:
  - Backends NumPy/CuPy (mesmo codigo em CPU e GPU), c64/c128 com
    verificacao cruzada integrada (fidelidade + TVD);
  - Fusao de portas 1q, caminho diagonal, amostragem vetorizada;
  - Observaveis Pauli arbitrarios (strings sobre IXYZ) e Z por qubit;
  - Vetores de Bloch por qubit e entropia de emaranhamento por qubit
    (particao 1 vs resto), para estudos de emaranhamento;
  - Ruido global, por porta e por qubit (depolarizante, amplitude
    damping, erro de leitura), com trajetorias estocasticas;
  - Trajetorias paralelas em multiplos processos (28 cores do Xeon);
  - Cancelamento cooperativo via callback should_cancel.

Convencao: little-endian (qubit 0 = bit menos significativo), como Qiskit.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import gates
from .backends import get_backend, to_cpu

MAX_QUBITS = 31
VERIFY_MAX_QUBITS = 26
PAULI_MAX_QUBITS = 26        # observavel Pauli usa uma copia do estado
BLOCH_MAX_QUBITS = 24
STATEVECTOR_MAX_QUBITS = 12
PROBS_FULL_MAX = 12
PROBS_TOP_K = 256
PARALLEL_MIN_SHOTS = 16

DTYPES = {
    "complex64": np.complex64,
    "c64": np.complex64,
    "complex128": np.complex128,
    "c128": np.complex128,
}

# Acelerador Numba (EXPERIMENTAL): kernels 1q paralelos em CPU.
# Ative com QSIM_USE_NUMBA=1; limiar de qubits em QSIM_NUMBA_MIN_QUBITS.
_USE_NUMBA = os.environ.get("QSIM_USE_NUMBA", "0") == "1"
_NUMBA_MIN_QUBITS = int(os.environ.get("QSIM_NUMBA_MIN_QUBITS", "18"))


class JobCancelled(Exception):
    """Job cancelado cooperativamente durante a execucao."""


# ---------------------------------------------------------------------------
# Modelo de ruido: global + por porta + por qubit
# ---------------------------------------------------------------------------
@dataclass
class NoiseModel:
    depolarizing: float = 0.0
    amplitude_damping: float = 0.0
    readout_error: float = 0.0
    per_gate: dict[str, dict[str, float]] = field(default_factory=dict)
    per_qubit: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return (self.depolarizing > 0 or self.amplitude_damping > 0
                or self.readout_error > 0 or bool(self.per_gate)
                or bool(self.per_qubit))

    def probs_for(self, gate: str, qubit: int) -> tuple[float, float]:
        """(depolarizante, amplitude_damping) efetivos para porta+qubit.
        Contribuicoes somadas e saturadas em 1.0."""
        dep = self.depolarizing
        damp = self.amplitude_damping
        g = self.per_gate.get(gate.lower())
        if g:
            dep += g.get("depolarizing", 0.0)
            damp += g.get("amplitude_damping", 0.0)
        q = self.per_qubit.get(qubit)
        if q:
            dep += q.get("depolarizing", 0.0)
            damp += q.get("amplitude_damping", 0.0)
        return min(dep, 1.0), min(damp, 1.0)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "NoiseModel":
        if not d:
            return cls()

        def _p(v: Any, name: str) -> float:
            f = float(v)
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"noise.{name} deve estar em [0, 1]")
            return f

        m = cls(
            depolarizing=_p(d.get("depolarizing", 0.0), "depolarizing"),
            amplitude_damping=_p(d.get("amplitude_damping", 0.0),
                                 "amplitude_damping"),
            readout_error=_p(d.get("readout_error", 0.0), "readout_error"),
        )
        for gate, spec in (d.get("per_gate") or {}).items():
            m.per_gate[str(gate).lower()] = {
                k: _p(v, f"per_gate.{gate}.{k}") for k, v in spec.items()
            }
        for qubit, spec in (d.get("per_qubit") or {}).items():
            m.per_qubit[int(qubit)] = {
                k: _p(v, f"per_qubit.{qubit}.{k}") for k, v in spec.items()
            }
        return m


@dataclass
class SimResult:
    counts: dict[str, int]
    shots: int
    num_qubits: int
    statevector: list[list[float]] | None = None
    probabilities: dict[str, float] | None = None
    expectation: dict[str, float] | None = None
    bloch: dict[str, list[float]] | None = None
    entanglement: dict[str, float] | None = None
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "shots": self.shots,
            "num_qubits": self.num_qubits,
            "statevector": self.statevector,
            "probabilities": self.probabilities,
            "expectation": self.expectation,
            "bloch": self.bloch,
            "entanglement": self.entanglement,
            "elapsed_ms": self.elapsed_ms,
            "metadata": self.metadata,
        }


class StatevectorSimulator:
    """Simulador exato por vetor de estado (CPU/GPU, c64/c128)."""

    def __init__(self, num_qubits: int, seed: int | None = None,
                 precision: str = "complex128", backend: str = "numpy"):
        if not isinstance(num_qubits, int) or not 1 <= num_qubits <= MAX_QUBITS:
            raise ValueError(
                f"num_qubits deve ser inteiro entre 1 e {MAX_QUBITS}"
            )
        if precision not in DTYPES:
            raise ValueError(f"Precisao invalida: '{precision}'")
        self.n = num_qubits
        self.dtype = DTYPES[precision]
        self.xp, self.backend_name = get_backend(backend)
        self.rng = np.random.default_rng(seed)
        self._numba = False
        if _USE_NUMBA and self.backend_name == "numpy":
            try:
                from . import numba_kernels
                self._numba = numba_kernels.HAS_NUMBA
            except Exception:
                self._numba = False
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = self.xp.zeros(2**self.n, dtype=self.dtype)
        self.state[0] = 1.0
        self.classical: dict[int, int] = {}

    def _tensor_view(self):
        return self.state.reshape([2] * self.n)

    # ------------------------------------------------------------------
    # Aplicacao de portas
    # ------------------------------------------------------------------
    def apply_matrix(self, mat: np.ndarray, targets: list[int]) -> None:
        xp = self.xp
        k = len(targets)
        if mat.shape != (2**k, 2**k):
            raise ValueError("Dimensao da matriz incompativel com os alvos")
        if len(set(targets)) != k:
            raise ValueError("Qubits alvo repetidos")
        for t in targets:
            if not isinstance(t, int) or not 0 <= t < self.n:
                raise ValueError(f"Qubit alvo fora do registro: {t}")

        # Caminho rapido EXPERIMENTAL: kernel Numba in-place para portas 1q.
        if self._numba and k == 1 and self.n >= _NUMBA_MIN_QUBITS:
            from . import numba_kernels as nk
            m = np.asarray(mat, dtype=np.complex128)
            nk.apply_1q(self.state, m[0, 0], m[0, 1], m[1, 0], m[1, 1],
                        targets[0], self.n)
            return

        mat = xp.asarray(mat, dtype=self.dtype)
        axes = [self.n - 1 - t for t in targets]
        psi = xp.moveaxis(self._tensor_view(), axes, range(k))

        diag_mat = xp.diag(xp.diag(mat))
        if int(xp.count_nonzero(mat - diag_mat)) == 0:
            d = xp.diag(mat).reshape([2] * k + [1] * (self.n - k))
            psi *= d
            return

        rest = psi.shape[k:]
        out = mat @ psi.reshape(2**k, -1)
        out = out.reshape([2] * k + list(rest))
        out = xp.moveaxis(out, range(k), axes)
        self.state = xp.ascontiguousarray(out.reshape(-1))

    def apply_gate(self, name: str, targets: list[int],
                   params: list[float] | None = None) -> None:
        mat, nq = gates.resolve(name, params)
        if len(targets) != nq:
            raise ValueError(
                f"Porta '{name}' atua em {nq} qubit(s), "
                f"recebeu {len(targets)} alvo(s)"
            )
        self.apply_matrix(mat, targets)

    # ------------------------------------------------------------------
    # Medicao, observaveis e diagnosticos de pesquisa
    # ------------------------------------------------------------------
    def _prob_one(self, qubit: int) -> float:
        xp = self.xp
        moved = xp.moveaxis(self._tensor_view(), self.n - 1 - qubit, 0)
        return float(to_cpu(xp.sum(xp.abs(moved[1]) ** 2)))

    def measure(self, qubit: int) -> int:
        xp = self.xp
        if not 0 <= qubit < self.n:
            raise ValueError(f"Qubit fora do registro: {qubit}")
        p1 = self._prob_one(qubit)
        outcome = int(self.rng.random() < p1)
        moved = xp.moveaxis(self._tensor_view(), self.n - 1 - qubit, 0)
        moved[1 - outcome] = 0
        norm = float(to_cpu(xp.linalg.norm(self.state)))
        if norm > 0:
            self.state /= norm
        self.classical[qubit] = outcome
        return outcome

    def expectation_z(self, qubit: int) -> float:
        return 1.0 - 2.0 * self._prob_one(qubit)

    def expectation_pauli(self, pauli: str) -> float:
        """<psi| P |psi> para P = string sobre IXYZ.

        Caractere na posicao j atua no qubit j (pauli[0] -> qubit 0).
        Custa uma copia do estado; limitado a PAULI_MAX_QUBITS.
        """
        xp = self.xp
        p = pauli.upper()
        if len(p) != self.n or set(p) - set("IXYZ"):
            raise ValueError(
                f"Pauli invalido: use string de tamanho {self.n} sobre IXYZ"
            )
        if self.n > PAULI_MAX_QUBITS:
            raise ValueError(
                f"Observavel Pauli limitado a {PAULI_MAX_QUBITS} qubits"
            )
        original = self.state
        self.state = original.copy()
        try:
            for q, ch in enumerate(p):
                if ch != "I":
                    self.apply_gate(ch.lower(), [q])
            value = complex(to_cpu(xp.vdot(original, self.state)))
        finally:
            self.state = original
        return float(value.real)

    def reduced_density_1q(self, qubit: int) -> np.ndarray:
        """Matriz densidade reduzida 2x2 do qubit (traco parcial do resto)."""
        xp = self.xp
        moved = xp.moveaxis(self._tensor_view(), self.n - 1 - qubit, 0)
        a0 = moved[0].reshape(-1)
        a1 = moved[1].reshape(-1)
        r00 = float(to_cpu(xp.sum(xp.abs(a0) ** 2)))
        r11 = float(to_cpu(xp.sum(xp.abs(a1) ** 2)))
        r01 = complex(to_cpu(xp.vdot(a1, a0)))  # sum a0 * conj(a1)
        return np.array([[r00, r01], [np.conj(r01), r11]],
                        dtype=np.complex128)

    def bloch_vector(self, qubit: int) -> tuple[float, float, float]:
        rho = self.reduced_density_1q(qubit)
        x = 2.0 * rho[0, 1].real
        y = -2.0 * rho[0, 1].imag
        z = (rho[0, 0] - rho[1, 1]).real
        return float(x), float(y), float(z)

    def entanglement_entropy(self, qubit: int) -> float:
        """Entropia de von Neumann (bits) da particao qubit vs resto.
        0 = separavel; 1 = maximamente emaranhado com o resto."""
        x, y, z = self.bloch_vector(qubit)
        r = min(math.sqrt(x * x + y * y + z * z), 1.0)
        lam = (1.0 + r) / 2.0
        ent = 0.0
        for p in (lam, 1.0 - lam):
            if p > 1e-15:
                ent -= p * math.log2(p)
        return float(ent)

    def probabilities(self):
        p = self.xp.abs(self.state) ** 2
        return p.astype(self.xp.float64, copy=False)

    # ------------------------------------------------------------------
    # Amostragem vetorizada
    # ------------------------------------------------------------------
    def sample(self, shots: int, readout_error: float = 0.0) -> dict[str, int]:
        xp = self.xp
        cdf = xp.cumsum(self.probabilities())
        cdf /= cdf[-1]
        u = xp.asarray(self.rng.random(shots))
        idx = xp.searchsorted(cdf, u, side="right")
        idx = xp.clip(idx, 0, 2**self.n - 1).astype(xp.int64)

        if readout_error > 0:
            flips = xp.asarray(self.rng.random((shots, self.n)) < readout_error)
            weights = xp.left_shift(
                xp.ones(self.n, dtype=xp.int64),
                xp.arange(self.n, dtype=xp.int64),
            )
            idx = xp.bitwise_xor(
                idx, (flips.astype(xp.int64) * weights).sum(axis=1)
            )

        vals, counts = xp.unique(idx, return_counts=True)
        vals, counts = to_cpu(vals), to_cpu(counts)
        return {
            format(int(v), f"0{self.n}b"): int(c)
            for v, c in zip(vals, counts)
        }

    # ------------------------------------------------------------------
    # Ruido estocastico (trajetoria)
    # ------------------------------------------------------------------
    def apply_noise(self, noise: NoiseModel, gate: str,
                    targets: list[int]) -> None:
        for q in targets:
            dep, damp = noise.probs_for(gate, q)
            if dep > 0 and self.rng.random() < dep:
                self.apply_gate(str(self.rng.choice(["x", "y", "z"])), [q])
            if damp > 0 and self.rng.random() < damp:
                if self.measure(q) == 1:
                    self.apply_gate("x", [q])


# ---------------------------------------------------------------------------
# Fabrica de simuladores: roteia backends especiais
# ---------------------------------------------------------------------------
def make_simulator(num_qubits: int, seed: int | None = None,
                   precision: str = "complex128",
                   backend: str = "numpy") -> StatevectorSimulator:
    if (backend or "").lower() == "custatevec":
        from .backends_custatevec import CuStateVecSimulator
        return CuStateVecSimulator(num_qubits, seed=seed, precision=precision)
    return StatevectorSimulator(num_qubits, seed=seed, precision=precision,
                                backend=backend)


# ---------------------------------------------------------------------------
# Compilacao: fusao de portas de 1 qubit
# ---------------------------------------------------------------------------
def compile_ops(ops: list[dict]) -> list[tuple[str, Any, list[int]]]:
    """Funde portas de 1 qubit consecutivas no mesmo qubit (mesmo separadas
    por portas em qubits disjuntos, que comutam)."""
    pending: dict[int, np.ndarray] = {}
    out: list[tuple[str, Any, list[int]]] = []

    def flush(qubits) -> None:
        for q in qubits:
            m = pending.pop(q, None)
            if m is not None:
                out.append(("u", m, [q]))

    for op in ops:
        if not isinstance(op, dict):
            raise ValueError("Operacao deve ser um objeto JSON")
        kind = op.get("op", "gate")
        if kind == "barrier":
            flush(list(pending.keys()))
            continue
        if kind in ("measure", "reset"):
            targets = [int(q) for q in op.get("targets", [])]
            if not targets:
                raise ValueError(f"Operacao '{kind}' sem alvos")
            flush(targets)
            out.append((kind, None, targets))
            continue
        if kind != "gate":
            raise ValueError(f"Operacao desconhecida: '{kind}'")
        name = op.get("gate")
        if not name:
            raise ValueError("Operacao de porta sem campo 'gate'")
        mat, nq = gates.resolve(name, op.get("params"))
        targets = [int(q) for q in op.get("targets", [])]
        if len(targets) != nq:
            raise ValueError(
                f"Porta '{name}' atua em {nq} qubit(s), "
                f"recebeu {len(targets)} alvo(s)"
            )
        if nq == 1:
            q = targets[0]
            pending[q] = mat @ pending[q] if q in pending else mat
        else:
            flush(targets)
            out.append(("u", mat, targets))

    flush(list(pending.keys()))
    return out


# ---------------------------------------------------------------------------
# Trajetorias (funcao de modulo para ser picklavel pelo ProcessPool)
# ---------------------------------------------------------------------------
def _run_trajectory_chunk(circuit: dict[str, Any], shots: int,
                          seed: int) -> dict[str, int]:
    n = int(circuit["num_qubits"])
    precision = str(circuit.get("precision", "complex128"))
    backend = str(circuit.get("backend", "numpy"))
    noise = NoiseModel.from_dict(circuit.get("noise"))
    ops = circuit["operations"]

    counts: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    for _ in range(shots):
        sim = make_simulator(
            n, seed=int(rng.integers(0, 2**63)),
            precision=precision, backend=backend,
        )
        for op in ops:
            kind = op.get("op", "gate")
            if kind == "barrier":
                continue
            if kind == "measure":
                for q in op["targets"]:
                    sim.measure(int(q))
                continue
            if kind == "reset":
                for q in op["targets"]:
                    if sim.measure(int(q)) == 1:
                        sim.apply_gate("x", [int(q)])
                continue
            targets = [int(q) for q in op["targets"]]
            sim.apply_gate(op["gate"], targets, op.get("params"))
            if noise.enabled:
                sim.apply_noise(noise, op["gate"], targets)
        for k, v in sim.sample(1, readout_error=noise.readout_error).items():
            counts[k] = counts.get(k, 0) + v
    return counts


# ---------------------------------------------------------------------------
# Execucao de circuitos JSON
# ---------------------------------------------------------------------------
def run_circuit(circuit: dict[str, Any],
                should_cancel: Callable[[], bool] | None = None) -> SimResult:
    """Executa um circuito no formato JSON do QSim.

    Campos: num_qubits, shots, seed, precision, backend, fuse, verify,
    parallel, return_statevector, return_bloch, return_entanglement,
    observables (["z0", "ZIXY..."]), noise (global/per_gate/per_qubit),
    operations.

    should_cancel: callback consultado periodicamente; se retornar True,
    a execucao para com JobCancelled (cancelamento cooperativo).
    """
    backend_req = str(circuit.get("backend", "numpy")).lower()
    if backend_req in ("aer", "aer-gpu"):
        from .backend_aer import run_aer
        return run_aer(circuit, should_cancel=should_cancel)

    t0 = time.perf_counter()
    cancel = should_cancel or (lambda: False)

    n = int(circuit["num_qubits"])
    shots = int(circuit.get("shots", 1024))
    seed = circuit.get("seed")
    ops = circuit.get("operations", [])
    precision = str(circuit.get("precision", "complex128"))
    backend = str(circuit.get("backend", "numpy"))
    fuse = bool(circuit.get("fuse", True))
    verify = bool(circuit.get("verify", False))
    parallel = bool(circuit.get("parallel", True))
    noise = NoiseModel.from_dict(circuit.get("noise"))
    want_sv = bool(circuit.get("return_statevector", False))
    want_bloch = bool(circuit.get("return_bloch", False))
    want_ent = bool(circuit.get("return_entanglement", False))
    observables = circuit.get("observables") or []

    if not isinstance(ops, list) or not ops:
        raise ValueError("Circuito sem operacoes")
    if shots < 1 or shots > 1_000_000:
        raise ValueError("shots deve estar entre 1 e 1000000")
    if precision not in DTYPES:
        raise ValueError(f"Precisao invalida: '{precision}'")

    has_mid = any(
        isinstance(o, dict) and o.get("op") in ("measure", "reset") for o in ops
    )

    # ------------------------------------------------------------------
    # Caminho deterministico
    # ------------------------------------------------------------------
    if not noise.enabled and not has_mid:
        compiled = compile_ops(ops) if fuse else None

        def run_once(prec: str) -> StatevectorSimulator:
            sim = make_simulator(n, seed=seed, precision=prec,
                                 backend=backend)
            if compiled is not None:
                for i, (_, mat, targets) in enumerate(compiled):
                    if i % 16 == 0 and cancel():
                        raise JobCancelled()
                    sim.apply_matrix(mat, targets)
            else:
                for i, op in enumerate(ops):
                    if i % 16 == 0 and cancel():
                        raise JobCancelled()
                    sim.apply_gate(op["gate"],
                                   [int(q) for q in op["targets"]],
                                   op.get("params"))
            return sim

        sim = run_once(precision)
        result = SimResult(
            counts=sim.sample(shots),
            shots=shots,
            num_qubits=n,
        )
        result.probabilities = _prob_summary(sim)
        if want_sv and n <= STATEVECTOR_MAX_QUBITS:
            sv = to_cpu(sim.state)
            result.statevector = [[float(a.real), float(a.imag)] for a in sv]
        if observables:
            result.expectation = _expectations(sim, observables)
        if want_bloch and n <= BLOCH_MAX_QUBITS:
            result.bloch = {
                str(q): [round(v, 10) for v in sim.bloch_vector(q)]
                for q in range(n)
            }
        if want_ent and n <= BLOCH_MAX_QUBITS:
            result.entanglement = {
                str(q): round(sim.entanglement_entropy(q), 10)
                for q in range(n)
            }

        meta: dict[str, Any] = {
            "mode": "exact",
            "precision": precision,
            "backend": sim.backend_name,
            "fused_ops": len(compiled) if compiled is not None else len(ops),
            "original_ops": len(ops),
        }

        if verify:
            if n > VERIFY_MAX_QUBITS:
                meta["verification"] = {
                    "skipped": f"verify limitado a {VERIFY_MAX_QUBITS} qubits"
                }
            else:
                other = ("complex128" if DTYPES[precision] == np.complex64
                         else "complex64")
                sim2 = run_once(other)
                a = to_cpu(sim.state).astype(np.complex128)
                b = to_cpu(sim2.state).astype(np.complex128)
                fidelity = float(np.abs(np.vdot(a, b)) ** 2)
                tvd = float(0.5 * np.sum(
                    np.abs(np.abs(a) ** 2 - np.abs(b) ** 2)
                ))
                meta["verification"] = {
                    "compared": f"{precision} vs {other}",
                    "state_fidelity": fidelity,
                    "total_variation_distance": tvd,
                    "passed": bool(fidelity > 1 - 1e-6 and tvd < 1e-4),
                }

        result.metadata = meta
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Caminho empirico: trajetorias estocasticas (paralelas em CPU)
    # ------------------------------------------------------------------
    _, backend_name = get_backend(backend)
    workers = int(os.environ.get("QSIM_TRAJECTORY_WORKERS", "0")) or \
        (os.cpu_count() or 1)
    use_parallel = (parallel and backend_name == "numpy"
                    and shots >= PARALLEL_MIN_SHOTS and workers > 1)

    base_rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}

    if use_parallel:
        workers = min(workers, shots)
        chunk = math.ceil(shots / workers)
        pieces = []
        remaining = shots
        while remaining > 0:
            take = min(chunk, remaining)
            pieces.append((take, int(base_rng.integers(0, 2**63))))
            remaining -= take
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_trajectory_chunk, circuit, take, s)
                for take, s in pieces
            ]
            for fut in as_completed(futures):
                if cancel():
                    for f in futures:
                        f.cancel()
                    raise JobCancelled()
                for k, v in fut.result().items():
                    counts[k] = counts.get(k, 0) + v
        mode_meta = {"parallel_workers": workers, "chunks": len(pieces)}
    else:
        done = 0
        while done < shots:
            if cancel():
                raise JobCancelled()
            take = min(64, shots - done)
            sub = _run_trajectory_chunk(
                circuit, take, int(base_rng.integers(0, 2**63))
            )
            for k, v in sub.items():
                counts[k] = counts.get(k, 0) + v
            done += take
        mode_meta = {"parallel_workers": 1}

    return SimResult(
        counts=dict(sorted(counts.items())),
        shots=shots,
        num_qubits=n,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metadata={
            "mode": "trajectories",
            "trajectories": shots,
            "precision": precision,
            **mode_meta,
        },
    )


def _prob_summary(sim: StatevectorSimulator) -> dict[str, float]:
    probs = to_cpu(sim.probabilities())
    n = sim.n
    if n <= PROBS_FULL_MAX:
        idx = np.nonzero(probs > 1e-12)[0]
    else:
        k = min(PROBS_TOP_K, probs.size)
        idx = np.argpartition(probs, -k)[-k:]
        idx = idx[probs[idx] > 1e-12]
    idx = idx[np.argsort(-probs[idx])]
    return {format(int(i), f"0{n}b"): float(probs[i]) for i in idx}


def _expectations(sim: StatevectorSimulator,
                  observables: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for obs in observables:
        key = str(obs).strip()
        low = key.lower()
        if low.startswith("z") and low[1:].isdigit():
            out[low] = sim.expectation_z(int(low[1:]))
        elif set(key.upper()) <= set("IXYZ") and len(key) == sim.n:
            out[key.upper()] = sim.expectation_pauli(key)
        else:
            raise ValueError(
                f"Observavel nao suportado: '{obs}' "
                f"(use z<idx> ou string Pauli de tamanho {sim.n})"
            )
    return out
