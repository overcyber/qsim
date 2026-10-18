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
# Kernels GPU in-place (P1.1): "auto" usa quando CuPy disponivel; "off"
# volta ao caminho generico moveaxis+matmul (referencia de comparacao).
_USE_KERNELS = os.environ.get("QSIM_KERNELS", "auto").lower()
# Amostragem por blocos (P0.3): CDF float64 materializada em pedacos de
# 16 Mi de amplitudes, permitindo amostrar 31 qubits sem dobrar a VRAM.
_SAMPLE_CHUNK = 1 << 24


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
        return self.quantum or self.readout_error > 0

    @property
    def quantum(self) -> bool:
        """Ha canais atuando sobre o estado quantico (exigem trajetorias ou
        um motor com canais). Erro de leitura sozinho nao conta: ele e
        aplicado na amostragem e cabe no caminho exato."""
        return (self.depolarizing > 0 or self.amplitude_damping > 0
                or any(v > 0 for spec in self.per_gate.values()
                       for v in spec.values())
                or any(v > 0 for spec in self.per_qubit.values()
                       for v in spec.values()))

    def probs_for(self, gate: str, qubit: int) -> tuple[float, float]:
        """(depolarizante, amplitude_damping) efetivos para porta+qubit.
        Contribuicoes somadas e saturadas em 1.0. O nome da porta e
        canonicalizado: cnot e cx recebem o mesmo ruido."""
        dep = self.depolarizing
        damp = self.amplitude_damping
        g = self.per_gate.get(gates.canonical(gate))
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
            if not gates.is_known(str(gate)):
                # antes era ignorado em silencio: ruido "configurado" que
                # nunca atuava
                raise ValueError(
                    f"noise.per_gate: porta desconhecida '{str(gate)[:16]}'"
                )
            entry = m.per_gate.setdefault(gates.canonical(str(gate)), {})
            for k, v in spec.items():
                # apelido e nome canonico no mesmo JSON se somam
                entry[k] = min(
                    entry.get(k, 0.0) + _p(v, f"per_gate.{gate}.{k}"), 1.0
                )
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
        # Kernels in-place (GPU): compilados sob demanda por dtype e
        # cacheados; qualquer falha desativa sem quebrar o worker.
        self._kernels = None
        if (self.backend_name == "cupy" and _USE_KERNELS in ("auto", "on", "1")):
            try:
                from .gpu_kernels import GpuKernels
                self._kernels = GpuKernels(self.xp, self.dtype)
            except Exception:
                self._kernels = None
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
            raise ValueError("Qubit alvo repetido")
        for t in targets:
            if not isinstance(t, int) or not 0 <= t < self.n:
                raise ValueError(f"Qubit alvo fora do registro: {t}")

        # Caminho rapido GPU (P1.1): kernels in-place, zero alocacao do
        # tamanho do estado — viabiliza 30-31 qubits na VRAM e, com a
        # indexacao O(1), domina o matmul generico em todas as faixas
        # (benchmark P40: 984 portas/s a 24q vs ~90 do caminho antigo).
        if self._kernels is not None:
            self._kernels.apply(self.state, mat, targets, self.n)
            return

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
        # Escrita de volta ATRAVES da visao moveaxis: muta o buffer original
        # in-place e elimina a copia final ascontiguousarray (que custava um
        # buffer adicional do tamanho do estado por porta).
        psi[...] = out

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
        Com kernels GPU (P1.3): uma passada sobre o estado, SEM copia e
        sem aplicar portas — teto pratico sobe para o do statevector.
        No caminho CPU, custa uma copia do estado (limitado a
        PAULI_MAX_QUBITS).
        """
        xp = self.xp
        p = pauli.upper()
        if len(p) != self.n or set(p) - set("IXYZ"):
            raise ValueError(
                f"Pauli invalido: use string de tamanho {self.n} sobre IXYZ"
            )
        if self._kernels is not None:
            return self._kernels.pauli_expectation(self.state, p, self.n)
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
        # dtype nativo (float32 para c64) e preenchimento por blocos em um
        # unico array pre-alocado: pico = estado + 1 saida + 1 bloco — nunca
        # 2x temporarios do tamanho do estado (inviavel em 31 qubits).
        xp = self.xp
        size = self.state.size
        dt = xp.float32 if self.dtype == np.complex64 else xp.float64
        out = xp.empty(size, dtype=dt)
        for s in range(0, size, _SAMPLE_CHUNK):
            chunk = self.state[s:s + _SAMPLE_CHUNK]
            out[s:s + _SAMPLE_CHUNK] = (chunk.real * chunk.real)
            out[s:s + _SAMPLE_CHUNK] += (chunk.imag * chunk.imag)
        return out

    # ------------------------------------------------------------------
    # Amostragem vetorizada (por blocos: P0.3)
    # ------------------------------------------------------------------
    def sample(self, shots: int, readout_error: float = 0.0) -> dict[str, int]:
        """Amostra `shots` resultados medindo todos os qubits.

        Nunca materializa um vetor de probabilidades do tamanho do estado:
        os quadrados |a|^2 sao computados por bloco direto do estado (duas
        passadas: total e busca), com acumulo float64 — viabiliza amostrar
        31 qubits c64 (8 GiB de estado) sem dobrar a VRAM em pico.
        """
        xp = self.xp
        size = self.state.size
        u = self.rng.random(shots)

        # passada 1: total (float64)
        total = 0.0
        for s in range(0, size, _SAMPLE_CHUNK):
            chunk = self.state[s:s + _SAMPLE_CHUNK]
            sq = (chunk.real * chunk.real) + (chunk.imag * chunk.imag)
            total += float(xp.sum(sq, dtype=xp.float64))

        targets = xp.asarray(u) * total
        idx = xp.empty(shots, dtype=xp.int64)
        filled = xp.zeros(shots, dtype=bool)
        offset = 0.0
        for s in range(0, size, _SAMPLE_CHUNK):
            chunk = self.state[s:s + _SAMPLE_CHUNK]
            sq = (chunk.real * chunk.real) + (chunk.imag * chunk.imag)
            cs = xp.cumsum(sq.astype(xp.float64)) + offset
            end = float(cs[-1])
            sel = (targets >= offset) & (targets < end)
            if bool(xp.any(sel)):
                pos = xp.searchsorted(cs, targets[sel], side="right")
                idx[sel] = xp.clip(pos, 0, chunk.size - 1) + s
                filled |= sel
            offset = end
        if not bool(xp.all(filled)):  # bordas de arredondamento: ultimo estado
            idx[~filled] = size - 1

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
                    targets: list[int]) -> bool:
        """Aplica o canal estocastico pos-porta. Retorna True se algum
        evento disparou (o estado divergiu do circuito limpo) — usado
        pelo reuso de trajetorias (P1.4)."""
        fired = False
        for q in targets:
            dep, damp = noise.probs_for(gate, q)
            if dep > 0 and self.rng.random() < dep:
                self.apply_gate(str(self.rng.choice(["x", "y", "z"])), [q])
                fired = True
            if damp > 0 and self.amplitude_damping(q, damp):
                fired = True
        return fired

    def amplitude_damping(self, qubit: int, gamma: float) -> bool:
        """Desdobramento em saltos quanticos do amplitude damping.

        Kraus: K0 = diag(1, sqrt(1-gamma)) e K1 = sqrt(gamma) |0><1|.
        O salto (K1) ocorre com probabilidade gamma * P(1); sem salto, o
        estado evolui por K0 e e renormalizado. A media sobre trajetorias
        reproduz exatamente o canal: p1 -> (1-gamma) p1 e coerencias
        multiplicadas por sqrt(1-gamma).

        Retorna True apenas quando ha salto. O ramo sem salto e
        deterministico (igual em todos os shots sem eventos), por isso nao
        quebra o reuso de trajetorias do P1.4.

        Correcao: a versao anterior sorteava gamma, fazia uma medicao
        projetiva e levava o qubit a |0>, o que e um canal de reset: as
        populacoes coincidiam, mas as coerencias decaiam por (1-gamma) em
        vez de sqrt(1-gamma).
        """
        p1 = self._prob_one(qubit)
        if p1 <= 0.0:
            return False
        moved = self.xp.moveaxis(self._tensor_view(), self.n - 1 - qubit, 0)
        if self.rng.random() < gamma * p1:
            moved[0] = moved[1]
            moved[1] = 0
            self.state /= math.sqrt(p1)
            return True
        moved[1] *= math.sqrt(1.0 - gamma)
        self.state /= math.sqrt(1.0 - gamma * p1)
        return False


# ---------------------------------------------------------------------------
# Fabrica de simuladores: roteia backends especiais
# ---------------------------------------------------------------------------
def make_simulator(num_qubits: int, seed: int | None = None,
                   precision: str = "complex128",
                   backend: str = "numpy") -> StatevectorSimulator:
    key = (backend or "").lower()
    if key == "custatevec":
        from .backends_custatevec import CuStateVecSimulator
        return CuStateVecSimulator(num_qubits, seed=seed, precision=precision)
    if key in ("auto", "cupy", "gpu"):
        # P1.2: em placas Volta+ (CC >= 7.0) o cuStateVec e o caminho padrao
        # (validado em tests/test_gpu.py); na Pascal (P40) segue CuPy com os
        # kernels in-place. Qualquer indisponibilidade cai no caminho generico.
        try:
            from .backends import available_backends
            if "custatevec" in available_backends():
                from .backends_custatevec import CuStateVecSimulator
                return CuStateVecSimulator(num_qubits, seed=seed,
                                           precision=precision)
        except Exception:
            pass
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
# Trajetorias (P1.4: reuso de prefixo deterministico + shots limpos em lote)
# ---------------------------------------------------------------------------
def _apply_static(sim: StatevectorSimulator, op: dict) -> None:
    """Aplica uma operacao SEM ruido e SEM medicao (prefixo deterministico)."""
    kind = op.get("op", "gate")
    if kind == "barrier":
        return
    sim.apply_gate(op["gate"], [int(q) for q in op["targets"]], op.get("params"))


def _trajectory_plan(circuit: dict) -> tuple[list[dict], list[dict], NoiseModel]:
    """Divide as operacoes em (prefixo deterministico, resto).

    O prefixo contem as portas que NUNCA divergem entre shots: aquelas
    antes da primeira cujo ruido pos-porta pode disparar, e antes de
    qualquer measure/reset intermediario. Executado 1 vez por chunk, o
    prefixo e clonado por shot — o custo dominante do caminho antigo
    (reexecutar tudo por shot) some para a parte deterministica.
    """
    noise = NoiseModel.from_dict(circuit.get("noise"))
    ops = circuit["operations"]
    split = 0
    for i, op in enumerate(ops):
        kind = op.get("op", "gate")
        if kind == "barrier":
            continue
        if kind in ("measure", "reset"):
            break
        if noise.enabled:
            for q in op["targets"]:
                if noise.probs_for(str(op.get("gate")), int(q)) != (0.0, 0.0):
                    # o ruido POS-PORTA desta porta e estocastico: a porta
                    # fica no resto (prefixo so porta cujo pos-ruido e zero —
                    # caso contrario o evento da fronteira se perde)
                    return ops[:split], ops[split:], noise
        split = i + 1
    return ops[:split], ops[split:], noise


# Tamanho fixo do bloco de shots com seed propria. Cada bloco recebe um
# filho de numpy.random.SeedSequence(seed), entao o resultado depende so da
# seed e do numero de shots, nunca do numero de workers nem do modo
# paralelo/sequencial (requisito de reprodutibilidade entre maquinas).
TRAJECTORY_BLOCK = 256


def _trajectory_blocks(shots: int, seed: int | None) -> list[tuple[int, Any]]:
    nblocks = math.ceil(shots / TRAJECTORY_BLOCK)
    children = np.random.SeedSequence(seed).spawn(nblocks)
    blocks = []
    remaining = shots
    for child in children:
        take = min(TRAJECTORY_BLOCK, remaining)
        blocks.append((take, child))
        remaining -= take
    return blocks


def _prefix_state(circuit: dict[str, Any]):
    """Estado apos o prefixo deterministico (sem aleatoriedade)."""
    prefix, _, _ = _trajectory_plan(circuit)
    sim = make_simulator(int(circuit["num_qubits"]), seed=0,
                         precision=str(circuit.get("precision", "complex128")),
                         backend=str(circuit.get("backend", "numpy")))
    for op in prefix:
        _apply_static(sim, op)
    return sim.state


def _run_trajectory_blocks(circuit: dict[str, Any],
                           blocks: list[tuple[int, Any]]) -> dict[str, int]:
    """Executa varios blocos num processo, calculando o prefixo uma vez."""
    pstate = _prefix_state(circuit)
    counts: dict[str, int] = {}
    for take, seed in blocks:
        sub = _run_trajectory_chunk(circuit, take, seed, prefix_state=pstate)
        for k, v in sub.items():
            counts[k] = counts.get(k, 0) + v
    return counts


def _run_trajectory_chunk(circuit: dict[str, Any], shots: int,
                          seed: Any, prefix_state=None) -> dict[str, int]:
    n = int(circuit["num_qubits"])
    precision = str(circuit.get("precision", "complex128"))
    backend = str(circuit.get("backend", "numpy"))
    prefix, rest, noise = _trajectory_plan(circuit)
    ro = noise.readout_error

    counts: dict[str, int] = {}
    rng = np.random.default_rng(seed)

    def flush(clean_sim, pending: int) -> None:
        if pending and clean_sim is not None:
            for k, v in clean_sim.sample(pending, readout_error=ro).items():
                counts[k] = counts.get(k, 0) + v

    # P1.4: prefixo deterministico uma unica vez; shots sem nenhum evento
    # estocastico (a esmagadora maioria com ruido leve) sao amostrados EM
    # LOTE do mesmo estado final, em vez de reexecutar o circuito.
    base = make_simulator(n, seed=int(rng.integers(0, 2**63)),
                          precision=precision, backend=backend)
    if prefix_state is not None:
        base.state = prefix_state.copy()
    else:
        for op in prefix:
            _apply_static(base, op)

    clean_pending = 0
    clean_sim = None
    for _ in range(shots):
        sim = make_simulator(n, seed=int(rng.integers(0, 2**63)),
                             precision=precision, backend=backend)
        sim.state = base.state.copy()
        diverged = False
        for op in rest:
            kind = op.get("op", "gate")
            if kind == "barrier":
                continue
            if kind == "measure":
                diverged = True
                for q in op["targets"]:
                    sim.measure(int(q))
                continue
            if kind == "reset":
                diverged = True
                for q in op["targets"]:
                    if sim.measure(int(q)) == 1:
                        sim.apply_gate("x", [int(q)])
                continue
            targets = [int(q) for q in op["targets"]]
            sim.apply_gate(op["gate"], targets, op.get("params"))
            if noise.enabled and sim.apply_noise(noise, op["gate"], targets):
                diverged = True
        if not diverged and not rest:
            # circuito inteiro deterministico: todos os shots saem do base
            clean_sim = base
            clean_pending += 1
            continue
        if not diverged:
            clean_sim = sim
            clean_pending += 1
            continue
        flush(clean_sim, clean_pending)
        clean_pending = 0
        for k, v in sim.sample(1, readout_error=ro).items():
            counts[k] = counts.get(k, 0) + v
    flush(clean_sim, clean_pending)
    return counts


# ---------------------------------------------------------------------------
# Resolucao do backend "auto"
# ---------------------------------------------------------------------------
def resolve_auto_backend(circuit: dict[str, Any]) -> str:
    """Escolhe o motor para backend "auto" entre os instalados.

    Circuitos com canais quanticos ou medicao intermediaria vao para o
    Aer (C++, canais nativos, matriz densidade em registros pequenos);
    trajetorias em GPU pelo nucleo Python seriam lentas. Circuitos exatos
    preferem aer-gpu, depois CuPy (kernels in-place do QSim), depois Aer em
    CPU e por fim o nucleo NumPy, que segue como oraculo de referencia.
    """
    from .backends import available_backends

    avail = set(available_backends())
    noise = NoiseModel.from_dict(circuit.get("noise"))
    has_mid = any(isinstance(o, dict) and o.get("op") in ("measure", "reset")
                  for o in circuit.get("operations", []))
    if noise.quantum or has_mid:
        order = ("aer-gpu", "aer", "numpy")
    else:
        order = ("aer-gpu", "cupy", "aer", "numpy")
    for name in order:
        if name in avail:
            return name
    return "numpy"


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
    if backend_req == "clifford":
        from .clifford import run_clifford
        return run_clifford(circuit, should_cancel=should_cancel)
    if backend_req in ("aer", "aer-gpu"):
        from .backend_aer import run_aer
        return run_aer(circuit, should_cancel=should_cancel)

    # Rota automatica (P2.1): circuito 100% Clifford acima de 20 qubits usa
    # tableau O(n^2) — o statevector seria inviavel ou desnecessariamente caro.
    if backend_req == "auto" and int(circuit.get("num_qubits", 0)) > 20:
        from .clifford import is_clifford_circuit, run_clifford
        noise = circuit.get("noise") or {}
        if (is_clifford_circuit(circuit.get("operations", []))
                and not any(noise.get(k) for k in
                            ("depolarizing", "amplitude_damping", "readout_error"))
                and not noise.get("per_gate") and not noise.get("per_qubit")
                and not (circuit.get("return_statevector")
                         or circuit.get("return_bloch")
                         or circuit.get("return_entanglement")
                         or circuit.get("observables"))):
            return run_clifford(circuit, should_cancel=should_cancel)

    # Demais casos de "auto": escolhe o melhor motor instalado (ver
    # resolve_auto_backend) e registra a escolha no resultado.
    if backend_req == "auto":
        resolved = resolve_auto_backend(circuit)
        result = run_circuit({**circuit, "backend": resolved},
                             should_cancel=should_cancel)
        result.metadata["backend_requested"] = "auto"
        result.metadata["backend_resolved"] = resolved
        return result

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
    if not noise.quantum and not has_mid:
        # Erro de leitura sozinho nao exige trajetorias: e aplicado na
        # amostragem a partir do estado exato.
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
            counts=sim.sample(shots, readout_error=noise.readout_error),
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
        if noise.readout_error > 0:
            meta["readout_error"] = noise.readout_error
            meta["note"] = ("counts incluem erro de leitura; probabilities "
                            "e observaveis sao do estado ideal")

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
    blocks = _trajectory_blocks(shots, seed)
    use_parallel = (parallel and backend_name == "numpy"
                    and shots >= PARALLEL_MIN_SHOTS and workers > 1
                    and len(blocks) > 1)

    counts: dict[str, int] = {}

    if use_parallel:
        workers = min(workers, len(blocks))
        # distribuicao round-robin dos blocos: o resultado nao depende dela
        tasks = [blocks[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_trajectory_blocks, circuit, task)
                for task in tasks
            ]
            for fut in as_completed(futures):
                if cancel():
                    for f in futures:
                        f.cancel()
                    raise JobCancelled()
                for k, v in fut.result().items():
                    counts[k] = counts.get(k, 0) + v
        mode_meta = {"parallel_workers": workers, "chunks": len(blocks)}
    else:
        pstate = _prefix_state(circuit)
        for take, block_seed in blocks:
            if cancel():
                raise JobCancelled()
            sub = _run_trajectory_chunk(circuit, take, block_seed,
                                        prefix_state=pstate)
            for k, v in sub.items():
                counts[k] = counts.get(k, 0) + v
        mode_meta = {"parallel_workers": 1, "chunks": len(blocks)}
    mode_meta["seed_scheme"] = (
        f"SeedSequence por bloco de {TRAJECTORY_BLOCK} shots"
    )

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
    n = sim.n
    if n <= PROBS_FULL_MAX:
        probs = to_cpu(sim.probabilities())
        idx = np.nonzero(probs > 1e-12)[0]
        idx = idx[np.argsort(-probs[idx])]
        return {format(int(i), f"0{n}b"): float(probs[i]) for i in idx}
    # n grande: top-k em STREAMING por blocos — nunca materializa o vetor
    # completo de probabilidades (em 30-31 qubits ele nao cabe junto do
    # estado na VRAM).
    xp = sim.xp
    k = min(PROBS_TOP_K, 2**n)
    best: dict[int, float] = {}
    size = sim.state.size
    for s in range(0, size, _SAMPLE_CHUNK):
        chunk = sim.state[s:s + _SAMPLE_CHUNK]
        sq = (chunk.real * chunk.real) + (chunk.imag * chunk.imag)
        kk = min(k, chunk.size)
        idx_l = xp.argpartition(sq, -kk)[-kk:]
        vals = to_cpu(sq[idx_l])
        idxs = to_cpu(idx_l)
        for i, v in zip(idxs, vals):
            if v > 1e-12:
                best[s + int(i)] = float(v)
    top = sorted(best.items(), key=lambda kv: -kv[1])[:k]
    return {format(i, f"0{n}b"): p for i, p in top}


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
