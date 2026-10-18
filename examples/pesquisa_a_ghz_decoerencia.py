"""Pesquisa A — Fragilidade do emaranhamento GHZ vs ruído (decoerência).

Linha: dinâmica de sistemas abertos. O GHZ-n perde a correlação global
exponencialmente com n. Oráculo: P(GHZ) ~ (1 - 2p/3)^(2(n-1)).

Uso (dentro da pasta qsim, com a stack no ar):
    docker run --rm --network host -v $PWD:/src:ro -w /src \
        -e PYTHONPATH=/src qsim-engine python3 examples/pesquisa_a_ghz_decoerencia.py
"""
import math

from sdk.qsim_client import run_sync

URL = "nats://127.0.0.1:4222"     # NATS local do qsim


def main():
    p = 0.02
    shots = 6000
    print(f"GHZ vs depolarizante p={p} (teorico: (1-2p/3)^(2(n-1)))")
    print("  n   P(GHZ) medido   teorico")
    for n in (2, 4, 6, 8, 10, 12):
        ops = [{"gate": "h", "targets": [0]}] + \
              [{"gate": "cx", "targets": [i, i + 1]} for i in range(n - 1)]
        payload = run_sync({"num_qubits": n, "shots": shots, "seed": 11,
                            "noise": {"depolarizing": p},
                            "operations": ops}, url=URL, timeout=120.0)
        assert payload["status"] == "done", payload
        counts = payload["result"]["counts"]
        good = sum(v for k, v in counts.items() if k in ("0" * n, "1" * n))
        theo = (1 - 2 * p / 3) ** (2 * (n - 1))
        print(f" {n:3d}   {good / shots:8.3f}       {theo:6.3f}")
    print("\nExtensões: variar p e ajustar a exponencial; trocar depolarizante"
          " por amplitude_damping; comparar GHZ vs estado produto |+>^n.")


if __name__ == "__main__":
    main()
