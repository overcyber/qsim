"""Pesquisa D: BB84 com interceptacao-reenvio (linha correlata: seguranca).

Pergunta: qual taxa de erro de bit (QBER) um espiao que intercepta uma
fracao f dos qubits provoca na chave peneirada, e a partir de que f ele e
detectado quando o canal ja tem ruido proprio?

Todo sorteio acontece DENTRO do circuito, em qubits auxiliares em |+>
(principio da medicao adiada): bit e base da Alice, base da Eve, base do
Bob e a decisao de interceptar. A medicao da Eve e um CNOT para um
registro dela. Sem ruido o circuito e unitario, entao a QBER sai exata das
probabilidades, sem erro de amostragem.

Oraculo sem ruido: QBER = f/4 (Eve acerta a base em metade dos casos;
quando erra, Bob erra o bit com probabilidade 1/2). O canal Eve -> Bob e
uma porta id com depolarizante p, que inverte o bit na base do Bob com
probabilidade n = 2p/3; como os dois erros se compoem por XOR,
QBER = e + n - 2en, com e = f/4. Limite de Shor-Preskill para BB84: QBER
acima de ~11% aborta a chave.

Qubits: q0 sinal | q1 bit da Alice | q2 base da Alice | q3 base da Eve |
        q4 base do Bob | q5 Eve intercepta? | q6 registro da Eve

Uso:
    python3 examples/pesquisa_d_bb84_interceptacao.py --local   # motor local
    python3 examples/pesquisa_d_bb84_interceptacao.py           # via NATS
"""
import math
import sys

URL = "nats://127.0.0.1:4222"
LOCAL = "--local" in sys.argv


def run(circuit: dict) -> dict:
    if LOCAL:
        from services.engine.simulator import run_circuit
        return run_circuit(circuit).to_dict()
    from sdk.qsim_client import run_sync
    payload = run_sync(circuit, url=URL, timeout=300.0)
    assert payload["status"] == "done", payload
    return payload["result"]


def ch(control: int, target: int) -> list[dict]:
    """Hadamard controlado: ry(pi/4) . CX . ry(-pi/4) no alvo."""
    return [{"gate": "ry", "targets": [target], "params": [math.pi / 4]},
            {"gate": "cx", "targets": [control, target]},
            {"gate": "ry", "targets": [target], "params": [-math.pi / 4]}]


def circuito(f: float) -> list[dict]:
    theta = 2 * math.asin(math.sqrt(f))          # P(q5 = 1) = f
    ops = [{"gate": "h", "targets": [q]} for q in (1, 2, 3, 4)]
    ops.append({"gate": "ry", "targets": [5], "params": [theta]})
    ops.append({"gate": "cx", "targets": [1, 0]})          # Alice codifica o bit
    ops += ch(2, 0)                                        # ... na base dela
    ops += ch(3, 0)                                        # Eve gira para a base dela
    ops.append({"gate": "ccx", "targets": [5, 0, 6]})      # mede (se interceptar)
    ops += ch(3, 0)                                        # reenvia na base dela
    ops.append({"gate": "id", "targets": [0]})             # canal ate o Bob
    ops += ch(4, 0)                                        # Bob mede na base dele
    return ops


def qber(dist: dict) -> tuple[float, float]:
    """(QBER na chave peneirada, fracao peneirada) de uma distribuicao."""
    total = sum(dist.values())
    sifted = err = 0.0
    for k, w in dist.items():
        q0, a, ba, bb = k[-1], k[-2], k[-3], k[-5]
        if ba == bb:
            sifted += w
            if q0 != a:
                err += w
    return err / sifted, sifted / total


def main():
    print("1) Exato (sem ruido): QBER x fracao interceptada f")
    print("    f     QBER     oraculo f/4   peneirada")
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = run({"num_qubits": 7, "shots": 1, "seed": 1, "backend": "auto",
                 "operations": circuito(f)})
        e, s = qber(r["probabilities"])
        print(f"  {f:4.2f}   {e:6.4f}   {f / 4:8.4f}      {s:5.3f}")

    p = 0.03
    n = 2 * p / 3
    shots = 40000
    print(f"\n2) Canal Eve->Bob com depolarizante p={p} (n = 2p/3 = {n:.3f}), "
          f"{shots} shots")
    print("    f     QBER     oraculo   sigma    decisao")
    for f in (0.0, 0.2, 0.3, 0.4, 0.5):
        r = run({"num_qubits": 7, "shots": shots, "seed": 7,
                 "backend": "auto",
                 "noise": {"per_gate": {"id": {"depolarizing": p}}},
                 "operations": circuito(f)})
        e, s = qber(r["counts"])
        teo = f / 4 + n - 2 * (f / 4) * n
        sigma = math.sqrt(teo * (1 - teo) / (s * shots))
        print(f"  {f:4.2f}   {e:6.4f}   {teo:6.4f}   {sigma:6.4f}   "
              f"{'aceita' if e < 0.11 else 'aborta (Eve detectada)'}")
    print(f"  fronteira teorica: f* = 4(0.11 - n)/(1 - 2n) = "
          f"{4 * (0.11 - n) / (1 - 2 * n):.3f}")

    print("\nLeitura: o ruido do canal cria um piso de QBER; abaixo da "
          "fronteira f* a Eve se esconde no ruido. Extensoes: varrer p e f "
          "e tracar a regiao de deteccao; trocar o canal por amplitude "
          "damping (assimetrico entre bases); ataque em base intermediaria "
          "(Breidbart).")


if __name__ == "__main__":
    main()
