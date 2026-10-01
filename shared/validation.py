"""Validacao estrutural de circuitos - defesa em profundidade.

Executada em DUAS camadas independentes:
  1. No gateway (Pydantic), protegendo a fronteira HTTP;
  2. No worker, antes de executar, protegendo contra payloads que cheguem
     ao NATS por outros caminhos (SDK, publicacao direta, componente
     comprometido). O worker nunca confia no que sai da fila.

Levanta ValueError com mensagem segura (sem eco de payload arbitrario).
"""
from __future__ import annotations

from typing import Any

MAX_QUBITS = 31
MAX_SHOTS = 1_000_000
MAX_OPERATIONS = 20_000
MAX_PARAMS = 8
MAX_LABEL_LEN = 128
ALLOWED_OPS = {"gate", "measure", "reset", "barrier"}
ALLOWED_PRECISIONS = {"complex64", "complex128", "c64", "c128"}
ALLOWED_BACKENDS = {"numpy", "cupy", "cpu", "gpu", "auto", "custatevec", "aer", "aer-gpu"}
ALLOWED_NOISE_KEYS = {"depolarizing", "amplitude_damping", "readout_error",
                      "per_gate", "per_qubit"}
ALLOWED_NOISE_SUBKEYS = {"depolarizing", "amplitude_damping"}
ALLOWED_CIRCUIT_KEYS = {
    "num_qubits", "shots", "seed", "precision", "backend", "fuse", "verify",
    "parallel", "return_statevector", "return_bloch", "return_entanglement",
    "observables", "noise", "operations",
}


def validate_circuit(circuit: Any) -> None:
    """Valida limites e estrutura de um circuito JSON. Nao executa nada."""
    if not isinstance(circuit, dict):
        raise ValueError("Circuito deve ser um objeto JSON")

    extra = set(circuit.keys()) - ALLOWED_CIRCUIT_KEYS
    if extra:
        raise ValueError(f"Campos desconhecidos no circuito: {sorted(extra)}")

    n = circuit.get("num_qubits")
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_QUBITS:
        raise ValueError(f"num_qubits deve ser inteiro entre 1 e {MAX_QUBITS}")

    shots = circuit.get("shots", 1024)
    if not isinstance(shots, int) or isinstance(shots, bool) or not 1 <= shots <= MAX_SHOTS:
        raise ValueError(f"shots deve ser inteiro entre 1 e {MAX_SHOTS}")

    seed = circuit.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed deve ser inteiro ou ausente")

    precision = circuit.get("precision", "complex128")
    if precision not in ALLOWED_PRECISIONS:
        raise ValueError("precision deve ser complex64 ou complex128")

    backend = circuit.get("backend", "numpy")
    if backend not in ALLOWED_BACKENDS:
        raise ValueError("backend deve ser numpy, cupy, custatevec, aer, aer-gpu ou auto")

    for flag in ("fuse", "verify", "parallel", "return_statevector",
                 "return_bloch", "return_entanglement"):
        v = circuit.get(flag)
        if v is not None and not isinstance(v, bool):
            raise ValueError(f"{flag} deve ser booleano")

    noise = circuit.get("noise")
    if noise is not None:
        _validate_noise(noise, n)

    observables = circuit.get("observables")
    if observables is not None:
        if not isinstance(observables, list) or len(observables) > MAX_QUBITS:
            raise ValueError("observables deve ser lista de ate 31 itens")
        for obs in observables:
            if not isinstance(obs, str) or not 1 <= len(obs) <= MAX_QUBITS + 1:
                raise ValueError("Observavel invalido")
            low = obs.lower()
            is_z = (low.startswith("z") and low[1:].isdigit()
                    and int(low[1:]) < n)
            is_pauli = (len(obs) == n and set(obs.upper()) <= set("IXYZ"))
            if not (is_z or is_pauli):
                raise ValueError(
                    "Observavel invalido (use z<idx> ou string Pauli "
                    "IXYZ de tamanho num_qubits)"
                )

    ops = circuit.get("operations")
    if not isinstance(ops, list) or not ops:
        raise ValueError("operations deve ser lista nao vazia")
    if len(ops) > MAX_OPERATIONS:
        raise ValueError(f"Circuito excede o maximo de {MAX_OPERATIONS} operacoes")

    for i, op in enumerate(ops):
        _validate_op(op, i, n)


def _validate_noise(noise: Any, n: int) -> None:
    if not isinstance(noise, dict):
        raise ValueError("noise deve ser um objeto")
    bad = set(noise.keys()) - ALLOWED_NOISE_KEYS
    if bad:
        raise ValueError(f"Chaves de ruido desconhecidas: {sorted(bad)}")

    def _prob(k: str, v: Any) -> None:
        if (not isinstance(v, (int, float)) or isinstance(v, bool)
                or not 0.0 <= float(v) <= 1.0):
            raise ValueError(f"noise.{k} deve ser numero em [0, 1]")

    for k in ("depolarizing", "amplitude_damping", "readout_error"):
        if k in noise:
            _prob(k, noise[k])

    per_gate = noise.get("per_gate")
    if per_gate is not None:
        if not isinstance(per_gate, dict) or len(per_gate) > 32:
            raise ValueError("noise.per_gate deve ser objeto com ate 32 portas")
        for gate, spec in per_gate.items():
            if (not isinstance(gate, str) or not gate.isalnum()
                    or len(gate) > 16 or not isinstance(spec, dict)):
                raise ValueError("noise.per_gate malformado")
            for k, v in spec.items():
                if k not in ALLOWED_NOISE_SUBKEYS:
                    raise ValueError(f"noise.per_gate.{gate}.{k} desconhecido")
                _prob(f"per_gate.{gate}.{k}", v)

    per_qubit = noise.get("per_qubit")
    if per_qubit is not None:
        if not isinstance(per_qubit, dict) or len(per_qubit) > MAX_QUBITS:
            raise ValueError("noise.per_qubit deve ser objeto com ate 31 qubits")
        for q, spec in per_qubit.items():
            if (not str(q).isdigit() or not int(q) < n
                    or not isinstance(spec, dict)):
                raise ValueError("noise.per_qubit malformado")
            for k, v in spec.items():
                if k not in ALLOWED_NOISE_SUBKEYS:
                    raise ValueError(f"noise.per_qubit.{q}.{k} desconhecido")
                _prob(f"per_qubit.{q}.{k}", v)


def _validate_op(op: Any, index: int, n: int) -> None:
    if not isinstance(op, dict):
        raise ValueError(f"Operacao {index} deve ser um objeto")
    kind = op.get("op", "gate")
    if kind not in ALLOWED_OPS:
        raise ValueError(f"Operacao {index}: tipo desconhecido")
    if kind == "barrier":
        return

    targets = op.get("targets")
    if (not isinstance(targets, list) or not targets
            or len(targets) > 3 or len(set(map(str, targets))) != len(targets)):
        raise ValueError(f"Operacao {index}: targets invalidos")
    for t in targets:
        if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < n:
            raise ValueError(f"Operacao {index}: alvo fora do registro")

    if kind in ("measure", "reset"):
        return

    gate = op.get("gate")
    if (not isinstance(gate, str) or not 1 <= len(gate) <= 16
            or not gate.isascii() or not gate.isalnum()):
        raise ValueError(f"Operacao {index}: nome de porta invalido")

    params = op.get("params")
    if params is not None:
        if not isinstance(params, list) or len(params) > MAX_PARAMS:
            raise ValueError(f"Operacao {index}: params invalidos")
        for p in params:
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                raise ValueError(f"Operacao {index}: parametro nao numerico")
            f = float(p)
            if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e6:
                raise ValueError(f"Operacao {index}: parametro fora de faixa")
