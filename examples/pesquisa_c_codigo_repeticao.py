"""Pesquisa C — Código de repetição: erro lógico vs físico (QEC).

Linha: correção de erros quântica. 1 qubit lógico em d físicos
(repetição contra bit-flip), canal depolarizante p numa camada de
identidades, decodificação por maioria (o sampler já mede todos os
qubits). Oráculo: erro binomial Bin(d, q), q = 2p/3 (só X/Y flipam o bit).

Uso: docker run --rm --network host -v $PWD:/src:ro -w /src \
        -e PYTHONPATH=/src qsim-engine python3 examples/pesquisa_c_codigo_repeticao.py
"""
import math

from sdk.qsim_client import run_sync

URL = "nats://127.0.0.1:4222"


def binomial_majority_error(d, q):
    return sum(math.comb(d, k) * q ** k * (1 - q) ** (d - k)
               for k in range(d // 2 + 1, d + 1))


def main():
    shots = 8000
    print("Taxa de erro lógico por distância (sem codificar = q = 2p/3)")
    print("  p     d=1     d=3(m/t)    d=5(m/t)    d=7(m/t)")
    for p in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25):
        q = 2 * p / 3
        row = f" {p:4.2f}  {q:6.3f}"
        for d in (3, 5, 7):
            ops = [{"gate": "i", "targets": [i]} for i in range(d)]
            payload = run_sync({"num_qubits": d, "shots": shots, "seed": 5,
                                "noise": {"depolarizing": p},
                                "operations": ops}, url=URL, timeout=200.0)
            assert payload["status"] == "done", payload
            counts = payload["result"]["counts"]
            err = sum(v for k, v in counts.items()
                      if k.count("1") > d // 2) / shots
            theo = binomial_majority_error(d, q)
            row += f"   {err:5.3f}/{theo:5.3f}"
        print(row)
    print("\nExtensões: decoder por síndrome (cx + medição intermediária em"
          " âncilas) em vez de maioria pós-hoc; código [[4,2,2]]; variar o"
          " canal (amplitude damping); escala Clifford para lógica de"
          " estabilizadores grande.")


if __name__ == "__main__":
    main()
