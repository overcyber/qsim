"""Testes do experimento do Tema 1 (canarios contra crosstalk).

Cobrem: concordancia das tres fontes de P(1), o teste binomial exato, o
CUSUM contra valores calculados a mao, a invariancia do detector
diferencial a deriva comum, a calibracao avaliada em dados independentes,
a reprodutibilidade por seed e o achado de que a referencia movel absorve
um ataque em rampa lenta.

Uso: PYTHONPATH=. python3 tests/test_tema1_canario.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.tema1_canario import benchmark_sequencial as bench
from experiments.tema1_canario.cenarios import (Cenario, amostrar,
                                                probabilidades,
                                                theta_por_janela)
from experiments.tema1_canario.detectores import (_cusum, calibrar_limiar,
                                                  cusum_diferencial,
                                                  primeiro_alarme)
from experiments.tema1_canario.modelo import (ModeloCanario, p1_analitico,
                                              p1_qsim_amostrado, p1_qsim_exato)
from experiments.tema1_canario.piloto_binomial import (cauda_superior,
                                                       teste_binomial_exato)


def test_tres_fontes_concordam() -> None:
    seed = 100
    for k in (1, 10):
        for theta in (0.0, 0.013, 0.05):
            for p in (0.0, 0.003, 0.02):
                a = p1_analitico(theta, k, p)
                e = p1_qsim_exato(theta, k, p)
                assert abs(a - e) < 1e-10, (k, theta, p, a, e)
                shots = 50_000
                s, _ = p1_qsim_amostrado(theta, k, p, shots, seed)
                seed += 1
                sigma = math.sqrt(max(a * (1 - a), 1e-12) / shots)
                assert abs(s - a) < 5 * sigma + 1e-12, (k, theta, p, s, a)
    print("[ok] P(1): formula fechada = matriz densidade do QSim (1e-10) e "
          "amostragem dentro de 5 sigma")


def test_teste_binomial_exato() -> None:
    s = cauda_superior(10, 0.5)
    assert abs(s[8] - (45 + 10 + 1) / 1024) < 1e-12
    assert abs(s[0] - 1.0) < 1e-12 and s[11] == 0.0
    for n, p0, p1 in ((500, 0.013, 0.023), (2000, 0.013, 0.023)):
        r = teste_binomial_exato(n, p0, p1, 0.01)
        assert r["tamanho_real"] <= 0.01
        assert cauda_superior(n, p0)[r["limiar_contagem"] - 1] > 0.01
    print("[ok] teste binomial exato: cauda correta e tamanho real <= alfa")


def test_cusum_valores_conhecidos() -> None:
    llr = np.array([[0.0, 1.0, -0.5, 2.0, -5.0, 1.5]])
    s = _cusum(llr, inicio=1)
    assert np.isneginf(s[0, 0])
    assert np.allclose(s[0, 1:], [1.0, 0.5, 2.5, 0.0, 1.5])
    assert primeiro_alarme(s, 2.0, 1)[0] == 3
    assert primeiro_alarme(s, 3.0, 1)[0] == -1
    print("[ok] CUSUM de Page e primeiro alarme conferem com o calculo a mao")


def test_diferencial_invariante_a_deriva_comum() -> None:
    rng = np.random.default_rng(1)
    for p in (0.01, 0.03, 0.08):
        c = rng.binomial(500, p, size=(2000, 60))
        f = rng.binomial(500, p, size=(2000, 60))
        llr_medio = np.mean(cusum_diferencial(c, 500, 0, 0.25, f)[:, 0])
        # sob H0 o incremento medio do LLR condicional e negativo (KL > 0)
        inc = np.mean(c * np.log(2 * 1.25 / 2.25) + f * np.log(2 / 2.25))
        assert inc < 0, (p, inc)
        assert llr_medio >= 0
    print("[ok] diferencial: incremento medio do LLR negativo sob H0 em "
          "qualquer nivel de ruido comum")


def _cfg_pequena() -> dict:
    cfg = dict(bench.CONFIG)
    cfg["replicas_calibracao"] = 2000
    cfg["replicas_teste"] = 2000
    return cfg


def test_calibracao_avaliada_em_dados_independentes() -> None:
    res = bench.executar(_cfg_pequena(), bench.CENARIOS[:1])
    for r in res["resultados"]:
        assert 0.02 <= r["falso_alarme"] <= 0.08, r
    print("[ok] falso alarme em S0 perto de 5% em replicas nao usadas na "
          "calibracao")


def test_reprodutibilidade_por_seed() -> None:
    cfg = _cfg_pequena()
    a = bench.executar(cfg, bench.CENARIOS[:2])
    b = bench.executar(cfg, bench.CENARIOS[:2])
    assert a["resultados"] == b["resultados"]
    assert not bench.verificar(a, b)
    print("[ok] benchmark identico com a mesma seed")


def test_rampa_lenta_absorvida_pela_referencia_movel() -> None:
    cen = [bench.CENARIOS[0], next(c for c in bench.CENARIOS if c.nome == "S7")]
    res = bench.executar(_cfg_pequena(), cen)
    det = {r["detector"]: r["deteccao"] for r in res["resultados"]
           if r["cenario"] == "S7"}
    assert det["cusum_movel"] < 0.5, det
    assert det["cusum_diferencial"] > 0.8, det
    print(f"[ok] rampa lenta: referencia movel detecta "
          f"{det['cusum_movel']:.2f}, diferencial {det['cusum_diferencial']:.2f}")


def test_perfis_de_ataque() -> None:
    W, tau = 300, 100
    rampa = theta_por_janela(Cenario("r", "", ataque="rampa", theta=0.03),
                             W, tau)
    assert rampa[tau - 1] == 0 and abs(rampa[-1] - 0.03) < 1e-12
    inter = theta_por_janela(Cenario("i", "", ataque="intermitente",
                                     theta=0.03, periodo=20,
                                     fracao_ativa=0.25), W, tau)
    assert abs(np.mean(inter[tau:] > 0) - 0.25) < 1e-12
    m = ModeloCanario(10, "analitico")
    p_prox, p_ref = probabilidades(Cenario("d", "", deriva=0.5, kappa=0.5),
                                   m, 0.002, W, tau)
    assert p_prox[-1] > p_ref[-1] > p_ref[0]
    c = amostrar(p_prox, 500, 3, np.random.default_rng(0))
    assert c.shape == (3, W)
    assert calibrar_limiar(np.zeros((10, W)), 50, 0.05) == 0.0
    print("[ok] perfis de ataque, deriva e descasamento conforme definidos")


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            fn()
    print("\nSuite do Tema 1: todos os testes passaram.")
