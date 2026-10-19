"""Testes do pipeline de calibracao real do Tema 1.

Cobrem: a conversao T1/T2 -> depolarizante equivalente contra a
fidelidade media do canal de relaxacao termica do Qiskit; a extracao de
dados de um backend simulado da IBM pelas duas APIs; a coleta retomavel e
deduplicada; a recuperacao do kappa verdadeiro em historico sintetico; e
o detector diferencial calibrado sob bases heterogeneas.

Uso: PYTHONPATH=. python3 tests/test_tema1_calibracao.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.tema1_canario.calibracao import (analisar_deriva,
                                                  benchmark_empirico,
                                                  coletar_ibm, sintetico)
from experiments.tema1_canario.calibracao.modelo_ruido import (
    fidelidade_media, p_idle, tau_para_p)
from experiments.tema1_canario.detectores import (cusum_diferencial,
                                                  cusum_diferencial_calibrado)

try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    HA_RUNTIME = True
except ImportError:
    HA_RUNTIME = False


def test_conversao_contra_relaxacao_termica() -> None:
    from qiskit.quantum_info import average_gate_fidelity
    from qiskit_aer.noise import thermal_relaxation_error

    for t1, t2, tau in ((200e-6, 150e-6, 500e-9), (80e-6, 160e-6, 2e-6),
                        (300e-6, 40e-6, 100e-9)):
        canal = thermal_relaxation_error(t1, t2, tau).to_quantumchannel()
        f_qiskit = average_gate_fidelity(canal)
        assert abs(f_qiskit - fidelidade_media(t1, t2, tau)) < 1e-12
        assert abs(p_idle(t1, t2, tau) - 1.5 * (1 - f_qiskit)) < 1e-12
    assert abs(p_idle(200e-6, 200e-6, tau_para_p(0.002, 200e-6, 200e-6))
               - 0.002) < 1e-12
    assert np.isnan(p_idle(0.0, 1e-4, 1e-7))
    print("[ok] conversao T1/T2: fidelidade media igual a do canal de "
          "relaxacao termica do Qiskit (1e-12)")


def test_extracao_backend_simulado() -> None:
    if not HA_RUNTIME:
        print("[pulado] qiskit-ibm-runtime ausente")
        return
    b = FakeTorino()
    a = coletar_ibm.extrair_de_properties(b.properties(), b.name, "x")
    t = coletar_ibm.extrair_de_target(b.target, b.name, "x")
    assert len(a) == len(t) == b.num_qubits
    for la, lt in zip(a[:20], t[:20]):
        assert abs(la["t1_s"] - lt["t1_s"]) < 1e-12
        assert abs(la["t2_s"] - lt["t2_s"]) < 1e-12
        assert abs(la["readout_error"] - lt["readout_error"]) < 1e-12
    assert a[0]["calibracao_utc"]
    print(f"[ok] extracao de {b.name}: properties e target concordam em "
          f"{b.num_qubits} qubits")


def test_coleta_retomavel_e_deduplicada() -> None:
    if not HA_RUNTIME:
        print("[pulado] qiskit-ibm-runtime ausente")
        return
    fake = FakeTorino()

    class Simulado:
        name = "simulado"
        num_qubits = fake.num_qubits
        coupling_map = fake.coupling_map
        chamadas = 0

        def properties(self, datetime=None):
            Simulado.chamadas += 1
            props = fake.properties()
            # muda a data de calibracao a cada 2 consultas
            props.last_update_date = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) \
                + dt.timedelta(days=Simulado.chamadas // 2)
            return props

    with tempfile.TemporaryDirectory() as tmp:
        original = coletar_ibm.DADOS
        coletar_ibm.DADOS = Path(tmp)
        try:
            coletar_ibm.salvar_mapa(Simulado(), "simulado")
            caminho = coletar_ibm.coletar(Simulado(), "simulado", dias=5,
                                          passo_horas=24, pausa_s=0)
            with caminho.open() as fh:
                linhas = list(csv.DictReader(fh))
            calib = {l["calibracao_utc"] for l in linhas}
            chamadas = Simulado.chamadas
            coletar_ibm.coletar(Simulado(), "simulado", dias=5,
                                passo_horas=24, pausa_s=0)
            assert Simulado.chamadas == chamadas, "retomada refez consultas"
            mapa = json.loads((Path(tmp) / "simulado_mapa.json").read_text())
        finally:
            coletar_ibm.DADOS = original
    assert len(linhas) == len(calib) * fake.num_qubits
    assert len(calib) < chamadas, "snapshots repetidos nao foram deduplicados"
    assert len(mapa["arestas"]) > 0
    print(f"[ok] coleta: {chamadas} consultas, {len(calib)} calibracoes "
          "distintas gravadas, retomada sem refazer consultas")


def test_recupera_kappa_sintetico() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        verdade = sintetico.gerar(Path(tmp))
        snaps, t1, t2 = analisar_deriva.carregar(Path(tmp) / "sintetico.csv")
        res = analisar_deriva.analisar(snaps, t1, t2, verdade["arestas"],
                                       reps_bootstrap=100)
    for faixa, beta in verdade["beta_teorico"].items():
        est = res["por_distancia"][faixa]["beta_mediano"]
        assert abs(est - beta) < 0.03, (faixa, est, beta)
    print("[ok] kappa por distancia recuperado do historico sintetico: " +
          ", ".join(f"d={f} {res['por_distancia'][f]['beta_mediano']:.3f} "
                    f"(verdade {b:.3f})" for f, b in verdade["beta_teorico"].items()))


def test_diferencial_calibrado_bases_heterogeneas() -> None:
    rng = np.random.default_rng(3)
    n, W = 500, 200
    c = rng.binomial(n, 0.02, size=(3000, W))     # canario proximo: base maior
    f = rng.binomial(n, 0.01, size=(3000, W))     # referencia: base menor
    s_orig = cusum_diferencial(c, n, 50, 0.25, f=f)
    s_cal = cusum_diferencial_calibrado(c, n, 50, 0.25, f=f)
    # sem ataque, o original acumula sem parar; o calibrado nao
    assert np.median(s_orig[:, -1]) > 50
    assert np.median(s_cal[:, -1]) < 5
    print(f"[ok] bases 0,02 contra 0,01 sem ataque: CUSUM diferencial "
          f"original termina em {np.median(s_orig[:, -1]):.0f}, calibrado em "
          f"{np.median(s_cal[:, -1]):.1f}")


def test_benchmark_empirico_ponta_a_ponta() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        verdade = sintetico.gerar(Path(tmp), dias=60)
        snaps, t1, t2 = analisar_deriva.carregar(Path(tmp) / "sintetico.csv")
        deriva = analisar_deriva.analisar(snaps, t1, t2, verdade["arestas"],
                                          reps_bootstrap=20)
    cfg = dict(benchmark_empirico.bench.CONFIG)
    cfg["replicas_calibracao"] = cfg["replicas_teste"] = 600
    a = benchmark_empirico.executar(deriva, cfg)
    b = benchmark_empirico.executar(deriva, cfg)
    assert a["resultados"] == b["resultados"]
    assert max(a["conferencia_fisica_max_dif"].values()) < 1e-10
    print("[ok] benchmark empirico: deterministico por seed e fisica "
          "conferida contra o QSim")


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            fn()
    print("\nSuite de calibracao do Tema 1: todos os testes passaram.")
