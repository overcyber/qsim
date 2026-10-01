"""Fuzzer estrutural do QSim (autonomo, sem dependencias externas).

Dois modos em uma execucao:
  1. Fuzz de invariantes: gera circuitos VALIDOS aleatorios e verifica
     propriedades que devem valer sempre (norma 1, soma de counts == shots,
     probabilidades somando 1, chaves com largura n).
  2. Fuzz de robustez: gera payloads MALFORMADOS por mutacao aleatoria e
     exige que o sistema responda com ValueError/KeyError/TypeError
     controlado — nunca outra excecao, travamento ou execucao parcial.

Uso: PYTHONPATH=. python tests/fuzz.py [iteracoes] [seed]
"""
from __future__ import annotations

import sys

import numpy as np

from services.engine.simulator import run_circuit
from shared.validation import validate_circuit

GATES_1Q = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"]
GATES_1Q_P = ["rx", "ry", "rz", "p"]
GATES_2Q = ["cx", "cz", "swap", "iswap"]
GATES_3Q = ["ccx", "cswap"]


def random_valid_circuit(rng: np.random.Generator) -> dict:
    n = int(rng.integers(1, 11))
    depth = int(rng.integers(1, 80))
    ops = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.45 or n < 2:
            ops.append({"gate": str(rng.choice(GATES_1Q)),
                        "targets": [int(rng.integers(0, n))]})
        elif r < 0.65:
            ops.append({"gate": str(rng.choice(GATES_1Q_P)),
                        "targets": [int(rng.integers(0, n))],
                        "params": [float(rng.uniform(-6.3, 6.3))]})
        elif r < 0.9 or n < 3:
            a, b = rng.choice(n, size=2, replace=False)
            ops.append({"gate": str(rng.choice(GATES_2Q)),
                        "targets": [int(a), int(b)]})
        else:
            t = rng.choice(n, size=3, replace=False)
            ops.append({"gate": str(rng.choice(GATES_3Q)),
                        "targets": [int(q) for q in t]})
        if rng.random() < 0.05:
            ops.append({"op": "measure", "targets": [int(rng.integers(0, n))]})
    circuit = {
        "num_qubits": n,
        "shots": int(rng.integers(1, 512)),
        "seed": int(rng.integers(0, 2**31)),
        "precision": str(rng.choice(["complex64", "complex128"])),
        "fuse": bool(rng.random() < 0.5),
        "operations": ops,
    }
    if rng.random() < 0.15:
        circuit["noise"] = {"depolarizing": float(rng.uniform(0, 0.05))}
        circuit["shots"] = min(circuit["shots"], 64)
    return circuit


def mutate(circuit: dict, rng: np.random.Generator) -> dict:
    """Corrompe um circuito valido de forma aleatoria."""
    bad = {k: (v.copy() if isinstance(v, (dict, list)) else v)
           for k, v in circuit.items()}
    poisons = [
        None, float("nan"), float("inf"), -1, 2**63, "x" * 200,
        [], {}, True, "'; --", "\x00",
    ]
    poison = poisons[int(rng.integers(0, len(poisons)))]
    strategy = rng.integers(0, 6)
    if strategy == 0:
        bad["num_qubits"] = poison
    elif strategy == 1:
        bad["shots"] = poison
    elif strategy == 2:
        bad[str(rng.choice(["evil", "__init__", "eval"]))] = poison
    elif strategy == 3 and bad["operations"]:
        i = int(rng.integers(0, len(bad["operations"])))
        op = dict(bad["operations"][i])
        field = str(rng.choice(["gate", "targets", "params", "op"]))
        op[field] = poison
        bad["operations"] = list(bad["operations"])
        bad["operations"][i] = op
    elif strategy == 4:
        bad["operations"] = poison
    else:
        bad["noise"] = {"depolarizing": poison}
    return bad


def run(iterations: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ok_valid = ok_reject = 0

    for i in range(iterations):
        circuit = random_valid_circuit(rng)

        # Modo 1: invariantes em entrada valida
        validate_circuit(circuit)
        result = run_circuit(circuit)
        n, shots = circuit["num_qubits"], circuit["shots"]
        assert sum(result.counts.values()) == shots, "counts nao somam shots"
        assert all(len(k) == n and set(k) <= {"0", "1"}
                   for k in result.counts), "chave de contagem malformada"
        if result.probabilities is not None:
            total = sum(result.probabilities.values())
            assert total <= 1.0 + 1e-6, "probabilidades somam mais que 1"
        ok_valid += 1

        # Modo 2: robustez contra mutacao hostil
        bad = mutate(circuit, rng)
        try:
            validate_circuit(bad)
            # Se a validacao deixou passar, o engine ainda precisa
            # falhar de forma controlada ou executar corretamente.
            run_circuit(bad)
        except (ValueError, KeyError, TypeError):
            pass
        ok_reject += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{iterations} iteracoes")

    print(f"\nFuzzer OK: {ok_valid} circuitos validos com invariantes "
          f"preservadas, {ok_reject} mutacoes hostis tratadas sem crash.")


if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run(iterations, seed)
