"""Backend Qiskit Aer: motor de execucao em C++ (producao).

O hot loop de aplicacao de portas roda no simulador do Aer (C++, AVX,
OpenMP; GPU com qiskit-aer-gpu). O nucleo Python do QSim permanece como
referencia auditavel e oraculo de verificacao. Todos os recursos do
pipeline sao preservados:

  - Precisoes complex64/complex128 e verificacao cruzada entre elas;
  - Amostragem com seed reprodutivel;
  - Observaveis Pauli, vetores de Bloch, entropia de emaranhamento e
    vetor de estado (diagnosticos calculados sobre o estado final);
  - Ruido global, por porta e por qubit, mapeado para o NoiseModel do
    Aer com o MESMO canal fisico do nucleo Python: nosso depolarizante
    (prob p de aplicar X/Y/Z uniforme) equivale ao depolarizing_error do
    Aer com parametro 4p/3; amplitude damping usa o canal de Kraus
    padrao; erro de leitura vira ReadoutError simetrico;
  - Medicao e reset no meio do circuito (nativos no Aer);
  - Cancelamento cooperativo em granularidade grossa (antes e depois da
    chamada ao motor, que e atomica em C++).

Selecao por job: "backend": "aer" (CPU) ou "aer-gpu" (device GPU do Aer,
requer qiskit-aer-gpu). Instalacao: pip install qiskit qiskit-aer.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from . import gates

GATE_SET_1Q = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "id"}
GATE_SET_1Q_P = {"rx", "ry", "rz", "p", "u"}
GATE_SET_2Q = {"cx", "cz", "swap"}
GATE_SET_2Q_P = {"cp", "rxx", "rzz"}
GATE_SET_3Q = {"ccx", "cswap"}

# Portas com instrucao nativa no Aer (chamadas como metodo do
# QuantumCircuit). Qualquer outra porta do QSim (ex.: iswap) e enviada
# como UnitaryGate com a matriz do proprio QSim e rotulo igual ao nome
# canonico; o rotulo e o que o NoiseModel do Aer usa para casar ruido.
AER_NATIVE = (GATE_SET_1Q | GATE_SET_1Q_P | GATE_SET_2Q | GATE_SET_2Q_P
              | GATE_SET_3Q)

# Ate este tamanho, circuitos com canais rodam por matriz densidade
# (uma evolucao + amostragem, sem trajetorias): 4^12 * 16 B = 256 MiB.
DENSITY_MATRIX_MAX_QUBITS = 12


def _require_aer():
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        return QuantumCircuit, AerSimulator
    except ImportError as err:
        raise RuntimeError(
            "Backend aer requer qiskit e qiskit-aer "
            f"(pip install qiskit qiskit-aer): {err}"
        )


@lru_cache(maxsize=1)
def aer_available() -> bool:
    try:
        _require_aer()
        return True
    except RuntimeError:
        return False


@lru_cache(maxsize=1)
def aer_gpu_available() -> bool:
    try:
        _, AerSimulator = _require_aer()
        return "GPU" in AerSimulator().available_devices()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Construcao do circuito Qiskit a partir do JSON do QSim
# ---------------------------------------------------------------------------
def build_qiskit_circuit(n: int, ops: list[dict], with_clbits: bool,
                         native: frozenset[str] | set[str] | None = None):
    """Traduz o circuito do QSim. `native` limita as portas enviadas como
    instrucao nativa as suportadas pelo metodo do Aer em uso; as demais
    vao como unitary rotulado (mesma matriz do QSim, mesmo ruido)."""
    QuantumCircuit, _ = _require_aer()
    native = AER_NATIVE if native is None else native
    qc = QuantumCircuit(n, n) if with_clbits else QuantumCircuit(n)
    for op in ops:
        kind = op.get("op", "gate")
        if kind == "barrier":
            qc.barrier()
            continue
        if kind == "measure":
            for q in op["targets"]:
                qc.measure(int(q), int(q))
            continue
        if kind == "reset":
            for q in op["targets"]:
                qc.reset(int(q))
            continue
        name = gates.canonical(op["gate"])
        targets = [int(q) for q in op["targets"]]
        params = op.get("params") or []
        if name in native:
            getattr(qc, name)(*params, *targets)
            continue
        mat, nq = gates.resolve(name, params)
        if nq != len(targets):
            raise ValueError(f"Porta '{name}' atua em {nq} qubit(s)")
        # Na matriz do QSim, targets[0] e o eixo mais significativo; no
        # Qiskit, qubits[0] e o menos significativo. Por isso a inversao.
        qc.unitary(mat, list(reversed(targets)), label=name)
    return qc


# ---------------------------------------------------------------------------
# Mapeamento do modelo de ruido do QSim para o NoiseModel do Aer
# ---------------------------------------------------------------------------
def _channel_1q(dep: float, damp: float, deph: float = 0.0):
    """Canal de 1 qubit equivalente ao do nucleo Python.

    Nosso depolarizante aplica X, Y ou Z com prob p/3 cada (nunca I). Isso
    e o depolarizing_error do Aer com parametro lambda = 4p/3, que o Aer
    aceita ate 4/3 (p = 1). Em seguida vem o amplitude damping e o phase
    damping, na mesma ordem do nucleo. O relaxamento termico ja chega aqui
    convertido nesses dois (NoiseModel.probs_for).
    """
    from qiskit_aer.noise import (amplitude_damping_error, depolarizing_error,
                                  phase_damping_error)
    error = None
    if dep > 0:
        error = depolarizing_error(4.0 * dep / 3.0, 1)
    if damp > 0:
        ad = amplitude_damping_error(damp)
        error = ad if error is None else error.compose(ad)
    if deph > 0:
        pd = phase_damping_error(deph)
        error = pd if error is None else error.compose(pd)
    return error


def build_noise_model(noise, n: int, ops: list[dict],
                      native: frozenset[str] | set[str] | None = None):
    """Traduz o NoiseModel do QSim para o do Aer.

    O ruido e registrado por instancia (porta canonica, qubits) que aparece
    no circuito, com o canal de 1 qubit de cada alvo em tensor, exatamente
    como o nucleo aplica. Assim valem global, per_gate e per_qubit para
    qualquer porta e qualquer aridade, inclusive apelidos e portas enviadas
    como unitary (iswap). A versao anterior so registrava uma lista fixa de
    portas: rzz, rxx e id ficavam sem ruido no Aer em silencio, e per_qubit
    com portas de 3 qubits era recusado.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    native = AER_NATIVE if native is None else native
    model = NoiseModel(basis_gates=sorted(AER_NATIVE | {"unitary"}))
    identity = depolarizing_error(0.0, 1)

    seen: set[tuple[str, tuple[int, ...]]] = set()
    for op in ops:
        if not isinstance(op, dict) or op.get("op", "gate") != "gate":
            continue
        name = gates.canonical(op["gate"])
        targets = [int(q) for q in op["targets"]]
        # O Aer casa ruido local pela tupla exata de qargs da instrucao.
        # Portas enviadas como unitary tem os alvos invertidos (ver
        # build_qiskit_circuit), entao o registro segue a mesma ordem.
        qargs = tuple(targets if name in native else reversed(targets))
        if (name, qargs) in seen:
            continue
        seen.add((name, qargs))
        channels = [_channel_1q(*noise.probs_for(name, q)) for q in qargs]
        if all(c is None for c in channels):
            continue
        # qubit i do erro corresponde a qargs[i]; tensor coloca o
        # operando da direita no qubit de menor indice
        error = channels[0] or identity
        for c in channels[1:]:
            error = (c or identity).tensor(error)
        model.add_quantum_error(error, name, list(qargs))

    if noise.readout_error > 0:
        p = noise.readout_error
        ro = ReadoutError([[1 - p, p], [p, 1 - p]])
        model.add_all_qubit_readout_error(ro)
    return model


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def native_gates_for(method: str) -> frozenset[str]:
    """Portas do QSim com instrucao nativa no metodo do Aer (o metodo
    density_matrix, por exemplo, nao aceita cswap)."""
    _, AerSimulator = _require_aer()
    basis = set(AerSimulator(method=method).configuration().basis_gates)
    return frozenset(AER_NATIVE & basis)


def aer_method_for(n: int, noisy: bool) -> str:
    return ("density_matrix" if noisy and n <= DENSITY_MATRIX_MAX_QUBITS
            else "statevector")


def _make_backend(device_gpu: bool, precision: str, noise_model=None,
                  n: int = 0):
    """Instancia o AerSimulator.

    Sem ruido: statevector. Com ruido: matriz densidade ate
    DENSITY_MATRIX_MAX_QUBITS (uma unica evolucao do canal, medido 14x a
    28x mais rapido que trajetorias para 2 a 10 qubits); acima disso,
    statevector com shot branching, que compartilha o estado entre shots
    ate o primeiro evento estocastico.
    """
    _, AerSimulator = _require_aer()
    kwargs: dict[str, Any] = {
        "method": "statevector",
        "precision": "single" if precision in ("complex64", "c64")
        else "double",
    }
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
        kwargs["method"] = aer_method_for(n, True)
        if kwargs["method"] == "statevector":
            kwargs["shot_branching_enable"] = True
    if device_gpu:
        if not aer_gpu_available():
            raise RuntimeError(
                "backend aer-gpu indisponivel: instale qiskit-aer-gpu e "
                "verifique a GPU (na P40 use backend cupy)"
            )
        kwargs["device"] = "GPU"
    return AerSimulator(**kwargs)


def aer_statevector(n: int, ops: list[dict], precision: str,
                    device_gpu: bool) -> np.ndarray:
    """Vetor de estado final calculado pelo motor C++/GPU do Aer."""
    qc = build_qiskit_circuit(n, ops, with_clbits=False)
    qc.save_statevector()
    backend = _make_backend(device_gpu, precision)
    result = backend.run(qc, shots=1).result()
    return np.asarray(result.get_statevector()).astype(np.complex128)


def aer_counts(n: int, ops: list[dict], shots: int, seed: int | None,
               precision: str, device_gpu: bool, noise) -> dict[str, int]:
    """Contagens com ruido, amostradas pelo motor do Aer.

    Semantica do nucleo QSim: medicoes intermediarias colapsam o estado e
    a amostragem final mede TODOS os qubits. Por isso o registro inteiro e
    medido ao final (sobrescrevendo os clbits intermediarios)."""
    native = native_gates_for(aer_method_for(n, noise.enabled))
    qc = build_qiskit_circuit(n, ops, with_clbits=True, native=native)
    qc.measure(range(n), range(n))
    noise_model = (build_noise_model(noise, n, ops, native=native)
                   if noise.enabled else None)
    backend = _make_backend(device_gpu, precision, noise_model=noise_model,
                            n=n)
    job = backend.run(qc, shots=shots,
                      seed_simulator=seed if seed is not None else None)
    raw = job.result().get_counts()
    return {k.replace(" ", ""): int(v) for k, v in
            sorted(raw.items())}


def run_aer(circuit: dict[str, Any],
            should_cancel: Callable[[], bool] | None = None):
    """Executa um circuito QSim no motor Aer, com paridade de recursos.

    Retorna SimResult identico ao do nucleo Python (mesmos campos e
    semantica), com metadata.engine = "aer".
    """
    from .simulator import (BLOCH_MAX_QUBITS, DTYPES, JobCancelled,
                            STATEVECTOR_MAX_QUBITS, VERIFY_MAX_QUBITS,
                            NoiseModel, SimResult, StatevectorSimulator,
                            _expectations, _prob_summary)

    t0 = time.perf_counter()
    cancel = should_cancel or (lambda: False)
    if cancel():
        raise JobCancelled()

    n = int(circuit["num_qubits"])
    shots = int(circuit.get("shots", 1024))
    seed = circuit.get("seed")
    ops = circuit.get("operations", [])
    precision = str(circuit.get("precision", "complex128"))
    device_gpu = str(circuit.get("backend", "aer")).lower() == "aer-gpu"
    verify = bool(circuit.get("verify", False))
    noise = NoiseModel.from_dict(circuit.get("noise"))
    want_sv = bool(circuit.get("return_statevector", False))
    want_bloch = bool(circuit.get("return_bloch", False))
    want_ent = bool(circuit.get("return_entanglement", False))
    observables = circuit.get("observables") or []

    has_mid = any(o.get("op") in ("measure", "reset") for o in ops
                  if isinstance(o, dict))

    # ------------------------------------------------------------------
    # Caminho empirico: ruido ou medicao intermediaria -> shots no Aer
    # ------------------------------------------------------------------
    if noise.quantum or has_mid:
        counts = aer_counts(n, ops, shots, seed, precision, device_gpu,
                            noise)
        if cancel():
            raise JobCancelled()
        return SimResult(
            counts=counts, shots=shots, num_qubits=n,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            metadata={
                "mode": "aer_sampling", "engine": "aer",
                "backend": "aer-gpu" if device_gpu else "aer",
                "aer_method": aer_method_for(n, noise.enabled),
                "device": "GPU" if device_gpu else "CPU",
                "precision": precision,
                "noise_channel_note": (
                    "depolarizante convertido por 4p/3 para equivaler ao "
                    "canal do nucleo Python" if noise.enabled else None
                ),
            },
        )

    # ------------------------------------------------------------------
    # Caminho deterministico: estado final no Aer, diagnosticos no nucleo
    # ------------------------------------------------------------------
    sv = aer_statevector(n, ops, precision, device_gpu)
    if cancel():
        raise JobCancelled()

    host = StatevectorSimulator(n, seed=seed, precision=precision,
                                backend="numpy")
    host.state = sv.astype(DTYPES[precision])

    # Erro de leitura sozinho: aplicado na amostragem a partir do estado
    # exato, igual ao nucleo (observaveis e probabilities seguem ideais).
    result = SimResult(counts=host.sample(shots,
                                          readout_error=noise.readout_error),
                       shots=shots, num_qubits=n)
    result.probabilities = _prob_summary(host)
    if want_sv and n <= STATEVECTOR_MAX_QUBITS:
        result.statevector = [[float(a.real), float(a.imag)] for a in sv]
    if observables:
        result.expectation = _expectations(host, observables)
    if want_bloch and n <= BLOCH_MAX_QUBITS:
        result.bloch = {str(q): [round(v, 10) for v in host.bloch_vector(q)]
                        for q in range(n)}
    if want_ent and n <= BLOCH_MAX_QUBITS:
        result.entanglement = {
            str(q): round(host.entanglement_entropy(q), 10)
            for q in range(n)
        }

    meta: dict[str, Any] = {
        "mode": "exact", "engine": "aer",
        "backend": "aer-gpu" if device_gpu else "aer",
        "device": "GPU" if device_gpu else "CPU",
        "precision": precision, "original_ops": len(ops),
    }
    if verify:
        if n > VERIFY_MAX_QUBITS:
            meta["verification"] = {
                "skipped": f"verify limitado a {VERIFY_MAX_QUBITS} qubits"
            }
        else:
            other = ("complex128" if precision in ("complex64", "c64")
                     else "complex64")
            sv2 = aer_statevector(n, ops, other, device_gpu)
            fidelity = float(np.abs(np.vdot(sv, sv2)) ** 2)
            tvd = float(0.5 * np.sum(np.abs(np.abs(sv) ** 2
                                            - np.abs(sv2) ** 2)))
            meta["verification"] = {
                "compared": f"{precision} vs {other}",
                "state_fidelity": fidelity,
                "total_variation_distance": tvd,
                "passed": bool(fidelity > 1 - 1e-6 and tvd < 1e-4),
            }
    result.metadata = meta
    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result
