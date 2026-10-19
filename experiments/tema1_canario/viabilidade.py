"""Verificacao de viabilidade do artigo 1 (canarios contra crosstalk).

Um unico comando que roda as checagens do experimento e emite um veredito
com criterios explicitos. Viabilidade aqui significa que a base
experimental se sustenta (fisica correta, resultado reproduzivel, sinal
detectavel) e que o problema continua aberto (nenhum detector avaliado
resolve todos os cenarios); nao e uma previsao de aceite.

Criterios:
  C1 fisica         formula fechada = matriz densidade do QSim (1e-10) e
                    amostragem pela API publica dentro de 5 sigma
  C2 reproducao     piloto e benchmark reproduzem as referencias gravadas
  C3 calibracao     falso alarme em S0 entre 2% e 8% para todo detector
                    (nominal 5%, avaliado em replicas independentes)
  C4 sinal          algum detector detecta o ataque constante (S3) com
                    probabilidade >= 0,8 e falso alarme previo <= 5%
  C5 lacuna         nenhum detector da linha de base (os da literatura:
                    Shewhart, CUSUM, CUSUM movel, diferenciais) e robusto
                    ao mesmo tempo a deriva comum (S1), deriva descasada
                    (S2) e rampa lenta (S7): falso alarme <= 10% em S1 e S2
                    e deteccao >= 0,8 em S7

Informativos (nao entram no veredito):
  C6 contribuicao   o detector proposto tem falso alarme <= 10% sob toda
                    deriva incoerente (S1, S2, S8, S9) e deteccao >= 0,8
                    nos ataques S3, S7 e S10 (atacante adaptativo)
  C7 limite         falso alarme sob deriva de frequencia (S11), um
                    deslocamento coerente de fase que nenhum canario de um
                    eixo distingue do ataque

Uso:
    PYTHONPATH=. python3 experiments/tema1_canario/viabilidade.py
"""
from __future__ import annotations

import json
import math
import sys
import time

from experiments.tema1_canario import benchmark_sequencial as bench
from experiments.tema1_canario import piloto_binomial as piloto
from experiments.tema1_canario.detectores import PROPOSTOS
from experiments.tema1_canario.manifesto import manifesto
from experiments.tema1_canario.modelo import (p1_analitico, p1_qsim_amostrado,
                                              p1_qsim_exato)


BASE = [d for d in bench.DETECTORES if d not in PROPOSTOS]


def c1_fisica() -> tuple[bool, str]:
    pior_dif, pior_z, seed = 0.0, 0.0, 5000
    for k in (1, 10, 60):
        for theta in (0.0, 0.013, 0.05):
            for p in (0.0, 0.003, 0.02):
                a = p1_analitico(theta, k, p)
                pior_dif = max(pior_dif, abs(a - p1_qsim_exato(theta, k, p)))
                s, _ = p1_qsim_amostrado(theta, k, p, 50_000, seed)
                seed += 1
                sig = math.sqrt(max(a * (1 - a), 1e-12) / 50_000)
                if a > 1e-9:
                    pior_z = max(pior_z, abs(s - a) / sig)
    ok = pior_dif < 1e-10 and pior_z < 5
    return ok, (f"maior diferenca formula/matriz densidade {pior_dif:.1e}; "
                f"maior desvio da amostragem {pior_z:.2f} sigma")


def c2_reproducao(res_piloto: dict, res_bench: dict) -> tuple[bool, str]:
    falhas = []
    if not piloto.SAIDA.exists() or not bench.SAIDA_JSON.exists():
        return False, "referencias gravadas ausentes em resultados/"
    falhas += piloto.verificar(json.loads(piloto.SAIDA.read_text()), res_piloto)
    falhas += bench.verificar(json.loads(bench.SAIDA_JSON.read_text()),
                              res_bench)
    if falhas:
        return False, "; ".join(falhas[:3])
    return True, "piloto e benchmark conferem com as referencias"


def _tabela(res_bench: dict) -> dict:
    return {(r["cenario"], r["detector"]): r for r in res_bench["resultados"]}


def c3_calibracao(t: dict) -> tuple[bool, str]:
    fa = {d: t[("S0", d)]["falso_alarme"] for d in bench.DETECTORES}
    ok = all(0.02 <= v <= 0.08 for v in fa.values())
    return ok, "falso alarme em S0: " + ", ".join(
        f"{d} {v:.3f}" for d, v in fa.items())


def c4_sinal(t: dict) -> tuple[bool, str]:
    melhores = sorted(
        ((t[("S3", d)]["deteccao"], d) for d in bench.DETECTORES
         if t[("S3", d)]["falso_alarme_pre"] <= 0.05), reverse=True)
    if not melhores:
        return False, "nenhum detector com falso alarme previo <= 5% em S3"
    det, nome = melhores[0]
    return det >= 0.8, f"melhor em S3: {nome} detecta {det:.3f}"


def c5_lacuna(t: dict) -> tuple[bool, str]:
    robustos = [d for d in BASE
                if t[("S1", d)]["falso_alarme"] <= 0.10
                and t[("S2", d)]["falso_alarme"] <= 0.10
                and t[("S7", d)]["deteccao"] >= 0.8]
    if robustos:
        return False, ("problema ja resolvido pelos detectores avaliados: "
                       + ", ".join(robustos))
    falhas = []
    for d in BASE:
        motivos = []
        if t[("S1", d)]["falso_alarme"] > 0.10:
            motivos.append(f"S1 FA {t[('S1', d)]['falso_alarme']:.2f}")
        if t[("S2", d)]["falso_alarme"] > 0.10:
            motivos.append(f"S2 FA {t[('S2', d)]['falso_alarme']:.2f}")
        if t[("S7", d)]["deteccao"] < 0.8:
            motivos.append(f"S7 det {t[('S7', d)]['deteccao']:.2f}")
        falhas.append(f"{d} ({', '.join(motivos)})")
    return True, "nenhum detector da linha de base robusto: " + "; ".join(falhas)


def c6_contribuicao(t: dict) -> tuple[bool, str]:
    partes, ok = [], True
    for d in PROPOSTOS:
        fa = {c: t[(c, d)]["falso_alarme"] for c in ("S1", "S2", "S8", "S9")}
        det = {c: t[(c, d)]["deteccao"] for c in ("S3", "S7", "S10")}
        ok &= all(v <= 0.10 for v in fa.values()) and \
            all(v >= 0.8 for v in det.values())
        partes.append(f"{d}: FA " + ", ".join(f"{c} {v:.3f}"
                                              for c, v in fa.items())
                      + "; det " + ", ".join(f"{c} {v:.3f}"
                                             for c, v in det.items()))
    return ok, "; ".join(partes)


def c7_limite(t: dict) -> tuple[bool, str]:
    fa = {d: t[("S11", d)]["falso_alarme"] for d in bench.DETECTORES}
    return all(v <= 0.10 for v in fa.values()), \
        "falso alarme sob deriva de frequencia (S11): " + ", ".join(
            f"{d} {v:.3f}" for d, v in fa.items())


def main() -> int:
    t0 = time.perf_counter()
    m = manifesto()
    print("Viabilidade do artigo 1: canarios contra injecao por crosstalk")
    print(f"commit {m['git']['commit'][:7]} ({m['git']['branch']}), "
          f"qiskit-aer {m['versoes']['qiskit_aer']}, numpy "
          f"{m['versoes']['numpy']}, {m['plataforma']}\n")

    print("executando piloto binomial (cerca de 30 s)...")
    res_piloto = piloto.executar()
    print("executando benchmark sequencial (cerca de 10 s)...\n")
    res_bench = bench.executar()
    t = _tabela(res_bench)

    criterios = [
        ("C1 fisica", c1_fisica()),
        ("C2 reproducao", c2_reproducao(res_piloto, res_bench)),
        ("C3 calibracao", c3_calibracao(t)),
        ("C4 sinal", c4_sinal(t)),
        ("C5 lacuna", c5_lacuna(t)),
    ]
    for nome, (ok, detalhe) in criterios:
        print(f"[{'ok' if ok else 'FALHOU'}] {nome}: {detalhe}")
    print("\ninformativos:")
    for nome, (ok, detalhe) in (("C6 contribuicao", c6_contribuicao(t)),
                                ("C7 limite", c7_limite(t))):
        print(f"[{'sim' if ok else 'nao'}] {nome}: {detalhe}")

    viavel = all(ok for _, (ok, _) in criterios)
    print(f"\ntempo total {time.perf_counter() - t0:.0f} s")
    print("\nVEREDITO:", "VIAVEL" if viavel else "NAO VIAVEL NESTE ESTADO")
    if viavel:
        print("A base experimental se sustenta e a linha de base nao resolve "
              "a deteccao robusta a deriva descasada e a atacante lento. Veja "
              "C6 e C7 para o detector proposto e o limite que permanece.")
    else:
        print("Veja os criterios que falharam acima antes de seguir.")
    return 0 if viavel else 1


if __name__ == "__main__":
    sys.exit(main())
