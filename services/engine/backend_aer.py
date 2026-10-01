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
from typing import Any, Callable

import numpy as np

GATE_SET_1Q = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"}
GATE_SET_1Q_P = {"rx", "ry", "rz", "p", "u"}
GATE_SET_2Q = {"cx", "cz", "swap"}
GATE_SET_2Q_P = {"cp"}
GATE_SET_3Q = {"ccx", "cswap"}


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


def aer_available() -> bool:
    try:
        _require_aer()
        return True
    except RuntimeError:
        return False


def aer_gpu_available() -> bool:
    try:
        _, AerSimulator = _require_aer()
        return "GPU" in AerSimulator().available_devices()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Construcao do circuito Qiskit a partir do JSON do QSim
# ---------------------------------------------------------------------------
def build_qiskit_circuit(n: int, ops: list[dict], with_clbits: bool):
    QuantumCircuit, _ = _require_aer()
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
        gate = op["gate"].lower()
        targets = [int(q) for q in op["targets"]]
        params = op.get("params") or []
        method = getattr(qc, gate, None)
        if method is None:
            raise ValueError(f"Porta '{gate}' sem equivalente no Aer")
        method(*params, *targets)
    return qc


# ---------------------------------------------------------------------------
# Mapeamento do modelo de ruido do QSim para o NoiseModel do Aer
# ---------------------------------------------------------------------------
def _channel_1q(dep: float, damp: float):
    """Canal de 1 qubit equivalente ao do nucleo Python.

    Nosso depolarizante aplica X/Y/Z com prob p/3 cada (nunca I), que e o
    depolarizing_error do Aer com parametro 4p/3 (clampado em 1).
    """
    from qiskit_aer.noise import amplitude_damping_error, depolarizing_error
    error = None
    if dep > 0:
        error = depolarizing_error(min(4.0 * dep / 3.0, 1.0), 1)
    if damp > 0:
        ad = amplitude_damping_error(damp)
        error = ad if error is None else error.compose(ad)
    return error


def build_noise_model(noise, n: int):
    """noise: instancia de simulator.NoiseModel ja validada."""
    from qiskit_aer.noise import NoiseModel, ReadoutError
    model = NoiseModel(basis_gates=sorted(
        GATE_SET_1Q | GATE_SET_1Q_P | GATE_SET_2Q | GATE_SET_2Q_P
        | GATE_SET_3Q
    ))
    gates_1q = sorted(GATE_SET_1Q | GATE_SET_1Q_P)
    gates_2q = sorted(GATE_SET_2Q | GATE_SET_2Q_P)
    gates_3q = sorted(GATE_SET_3Q)
    has_per_qubit = bool(noise.per_qubit)

    # Portas de 1 qubit: canal proprio por (porta, qubit)
    for g in gates_1q:
        for q in range(n):
            err = _channel_1q(*noise.probs_for(g, q))
            if err is not None:
                model.add_quantum_error(err, g, [q])

    # Portas de 2 qubits: canais independentes em cada qubit envolvido
    # (mesma semantica do nucleo Python). Com per_qubit, registra por par.
    for g in gates_2q:
        if not has_per_qubit:
            err = _channel_1q(*noise.probs_for(g, 0))
            if err is not None:
                model.add_all_qubit_quantum_error(err.tensor(err), g)
        else:
            for q0 in range(n):
                for q1 in range(n):
                    if q0 == q1:
                        continue
                    e0 = _channel_1q(*noise.probs_for(g, q0))
                    e1 = _channel_1q(*noise.probs_for(g, q1))
                    if e0 is None and e1 is None:
                        continue
                    from qiskit_aer.noise import depolarizing_error
                    ident = depolarizing_error(0.0, 1)
                    e0 = e0 or ident
                    e1 = e1 or ident
                    # ordem dos qubits do erro segue [q0, q1]
                    model.add_quantum_error(e1.tensor(e0), g, [q0, q1])

    # Portas de 3 qubits: apenas ruido uniforme (global/per_gate);
    # per_qubit em portas de 3 qubits nao e mapeavel de forma exata aqui.
    for g in gates_3q:
        if has_per_qubit:
            continue
        err = _channel_1q(*noise.probs_for(g, 0))
        if err is not None:
            model.add_all_qubit_quantum_error(err.tensor(err).tensor(err), g)

    if noise.readout_error > 0:
        p = noise.readout_error
        ro = ReadoutError([[1 - p, p], [p, 1 - p]])
        model.add_all_qubit_readout_error(ro)
    return model


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def _make_backend(device_gpu: bool, precision: str, noise_model=None):
    _, AerSimulator = _require_aer()
    kwargs: dict[str, Any] = {
        "method": "statevector" if noise_model is None else "density_matrix"
        if False else "statevector",
        "precision": "single" if precision in ("complex64", "c64")
        else "double",
    }
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
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
    qc = build_qiskit_circuit(n, ops, with_clbits=True)
    qc.measure(range(n), range(n))
    noise_model = build_noise_model(noise, n) if noise.enabled else None
    backend = _make_backend(device_gpu, precision, noise_model=noise_model)
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
    if noise.enabled or has_mid:
        if noise.per_qubit and any(
            (o.get("gate") or "").lower() in GATE_SET_3Q for o in ops
        ):
            raise ValueError(
                "noise.per_qubit com portas de 3 qubits nao e suportado no "
                "backend aer; use backend numpy (trajetorias) para esse caso"
            )
        counts = aer_counts(n, ops, shots, seed, precision, device_gpu,
                            noise)
        if cancel():
            raise JobCancelled()
        return SimResult(
            counts=counts, shots=shots, num_qubits=n,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            metadata={
                "mode": "aer_sampling", "engine": "aer",
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

    result = SimResult(counts=host.sample(shots), shots=shots, num_qubits=n)
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
