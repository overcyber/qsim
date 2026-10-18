"""Pesquisa E: teleporte quantico com memoria ruidosa (linha relacionada:
protocolos e redes quanticas).

Pergunta: ate que nivel de ruido no par emaranhado o teleporte supera o
melhor protocolo classico (medir e reenviar, fidelidade media 2/3)?

O QSim nao tem controle classico por resultado de medicao (feed-forward),
mas nao precisa: pelo principio da medicao adiada, as correcoes X e Z
condicionadas aos bits da Alice viram CX(q1->q2) e CZ(q0->q2) antes de
qualquer medicao. O circuito fica unitario e os valores esperados no qubit
de destino saem exatos.

Ruido: porta id com depolarizante p em q1 e q2 logo apos criar o par
(armazenamento/distribuicao). Cada canal atua no estado teleportado como
um depolarizante de mesma intensidade, entao o vetor de Bloch encolhe por
(1 - 4p/3)^2 e a fidelidade media e F = (1 + (1 - 4p/3)^2) / 2.
F > 2/3 exige p < 3(1 - 1/sqrt(3))/4 = 0,317.

Uso:
    python3 examples/pesquisa_e_teletransporte.py --local
    python3 examples/pesquisa_e_teletransporte.py
"""
import math
import sys

URL = "nats://127.0.0.1:4222"
LOCAL = "--local" in sys.argv
ENTRADAS = [(0.0, 0.0), (math.pi / 2, 0.0), (math.pi / 2, math.pi / 2),
            (2.0, 1.0), (1.0, 4.0), (math.pi, 0.0)]


def run(circuit: dict) -> dict:
    if LOCAL:
        from services.engine.simulator import run_circuit
        return run_circuit(circuit).to_dict()
    from sdk.qsim_client import run_sync
    payload = run_sync(circuit, url=URL, timeout=300.0)
    assert payload["status"] == "done", payload
    return payload["result"]


def teleporte(theta: float, phi: float, ruido: bool) -> list[dict]:
    ops = [{"gate": "ry", "targets": [0], "params": [theta]},   # |psi> em q0
           {"gate": "rz", "targets": [0], "params": [phi]},
           {"gate": "h", "targets": [1]},                       # par de Bell
           {"gate": "cx", "targets": [1, 2]}]
    if ruido:
        ops += [{"gate": "id", "targets": [1]}, {"gate": "id", "targets": [2]}]
    ops += [{"gate": "cx", "targets": [0, 1]},                  # Alice
            {"gate": "h", "targets": [0]},
            {"gate": "cx", "targets": [1, 2]},                  # correcoes
            {"gate": "cz", "targets": [0, 2]}]                  # adiadas
    return ops


def alvo(theta: float, phi: float) -> tuple[float, float, float]:
    return (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
            math.cos(theta))


def bloch_medido(theta: float, phi: float, p: float, shots: int):
    """Tomografia do qubit 2 por contagens nas bases X, Y e Z."""
    rotacoes = {"x": [{"gate": "h", "targets": [2]}],
                "y": [{"gate": "sdg", "targets": [2]},
                      {"gate": "h", "targets": [2]}],
                "z": []}
    r = []
    for base in ("x", "y", "z"):
        res = run({"num_qubits": 3, "shots": shots, "seed": 5,
                   "backend": "auto",
                   "noise": {"per_gate": {"id": {"depolarizing": p}}},
                   "operations": teleporte(theta, phi, True) + rotacoes[base]})
        c = res["counts"]
        p0 = sum(v for k, v in c.items() if k[0] == "0") / shots   # k[0] = q2
        r.append(2 * p0 - 1)
    return r


def main():
    print("1) Sem ruido, valores esperados exatos no qubit de destino")
    for theta, phi in ENTRADAS:
        res = run({"num_qubits": 3, "shots": 1, "seed": 1, "backend": "auto",
                   "observables": ["IIX", "IIY", "IIZ"],
                   "operations": teleporte(theta, phi, False)})
        e = res["expectation"]
        r = (e["IIX"], e["IIY"], e["IIZ"])
        t = alvo(theta, phi)
        f = (1 + sum(a * b for a, b in zip(r, t))) / 2
        print(f"  theta={theta:5.3f} phi={phi:5.3f}  fidelidade = {f:.12f}")

    shots = 20000
    print(f"\n2) Memoria ruidosa: fidelidade media sobre {len(ENTRADAS)} "
          f"estados ({shots} shots por base)")
    print("    p      F medida   oraculo   supera 2/3?")
    for p in (0.0, 0.1, 0.2, 0.3, 0.35):
        fs = []
        for theta, phi in ENTRADAS:
            r = bloch_medido(theta, phi, p, shots)
            t = alvo(theta, phi)
            fs.append((1 + sum(a * b for a, b in zip(r, t))) / 2)
        f_med = sum(fs) / len(fs)
        teo = (1 + (1 - 4 * p / 3) ** 2) / 2
        print(f"  {p:4.2f}    {f_med:7.4f}   {teo:7.4f}   "
              f"{'sim' if f_med > 2 / 3 else 'nao'}")
    print(f"  fronteira teorica: p* = {3 * (1 - 1 / math.sqrt(3)) / 4:.3f}")

    print("\nExtensoes: amplitude damping na memoria (canal nao unital: a "
          "fidelidade passa a depender do estado de entrada); purificacao "
          "de emaranhamento antes do teleporte; troca de emaranhamento "
          "(entanglement swapping) em cadeia, base de repetidores.")


if __name__ == "__main__":
    main()
