"""Coletor do historico de calibracao de um backend da IBM.

Para cada instante de uma grade (padrao: um por dia nos ultimos 90 dias),
consulta a calibracao vigente naquele momento e grava, por qubit: T1, T2,
erro de leitura, erro da porta sx e a data da calibracao. Tambem grava o
mapa de acoplamento, necessario para medir distancias no chip.

Requer conta na IBM Quantum Platform (o plano aberto basta) e o pacote
qiskit-ibm-runtime. A conta e lida da configuracao salva do Qiskit ou das
variaveis QISKIT_IBM_TOKEN e QISKIT_IBM_INSTANCE (ver --ajuda).

Uso:
    pip install qiskit-ibm-runtime
    PYTHONPATH=. python3 experiments/tema1_canario/calibracao/coletar_ibm.py \\
        --backend ibm_torino --dias 90

Saida (em experiments/tema1_canario/calibracao/dados/):
    <backend>.csv          uma linha por (snapshot de calibracao, qubit)
    <backend>_mapa.json    arestas do mapa de acoplamento e metadados
    <backend>_consultas.log  instantes ja consultados (para retomar)

A coleta e retomavel: instantes ja consultados sao pulados, e snapshots
com a mesma data de calibracao sao gravados uma vez so.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

DADOS = Path(__file__).parent / "dados"
CAMPOS = ["backend", "consulta_utc", "calibracao_utc", "qubit", "t1_s",
          "t2_s", "readout_error", "sx_error"]


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def extrair_de_properties(props, backend: str, consulta: str) -> list[dict]:
    """Linhas a partir de um BackendProperties (API classica)."""
    linhas = []
    cal = props.last_update_date
    cal_iso = cal.astimezone(dt.timezone.utc).isoformat() if cal else ""
    for q in range(len(props.qubits)):
        qp = props.qubit_property(q)

        def val(nome):
            item = qp.get(nome)
            return _num(item[0] if isinstance(item, tuple) else item)

        try:
            sx = _num(props.gate_error("sx", q))
        except Exception:
            sx = None
        linhas.append({"backend": backend, "consulta_utc": consulta,
                       "calibracao_utc": cal_iso, "qubit": q,
                       "t1_s": val("T1"), "t2_s": val("T2"),
                       "readout_error": val("readout_error"),
                       "sx_error": sx})
    return linhas


def extrair_de_target(target, backend: str, consulta: str,
                      calibracao: str = "") -> list[dict]:
    """Linhas a partir de um Target (API nova, target_history)."""
    linhas = []
    for q in range(target.num_qubits):
        qp = target.qubit_properties[q] if target.qubit_properties else None

        def erro(op):
            try:
                return _num(target[op][(q,)].error)
            except Exception:
                return None

        linhas.append({"backend": backend, "consulta_utc": consulta,
                       "calibracao_utc": calibracao, "qubit": q,
                       "t1_s": _num(getattr(qp, "t1", None)),
                       "t2_s": _num(getattr(qp, "t2", None)),
                       "readout_error": erro("measure"),
                       "sx_error": erro("sx")})
    return linhas


def consultar(backend, instante: dt.datetime, nome: str) -> list[dict]:
    consulta = instante.isoformat()
    try:
        props = backend.properties(datetime=instante)
        if props is not None:
            return extrair_de_properties(props, nome, consulta)
    except NotImplementedError:
        pass
    target = backend.target_history(datetime=instante)
    return extrair_de_target(target, nome, consulta)


def _ja_coletados(caminho: Path, log: Path) -> tuple[set[str], set[str]]:
    """Instantes ja consultados (inclusive os de calibracao repetida, que
    nao geram linhas no CSV) e datas de calibracao ja gravadas."""
    consultas, calibracoes = set(), set()
    if log.exists():
        consultas.update(l.strip() for l in log.read_text().splitlines()
                         if l.strip())
    if caminho.exists():
        with caminho.open() as fh:
            for linha in csv.DictReader(fh):
                consultas.add(linha["consulta_utc"])
                if linha["calibracao_utc"]:
                    calibracoes.add(linha["calibracao_utc"])
    return consultas, calibracoes


def salvar_mapa(backend, nome: str) -> Path:
    DADOS.mkdir(parents=True, exist_ok=True)
    arestas = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    caminho = DADOS / f"{nome}_mapa.json"
    caminho.write_text(json.dumps({
        "backend": nome, "num_qubits": backend.num_qubits,
        "arestas": [list(a) for a in arestas],
        "coletado_em_utc": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
    }, indent=1))
    return caminho


def coletar(backend, nome: str, dias: int, passo_horas: float,
            pausa_s: float = 0.5) -> Path:
    DADOS.mkdir(parents=True, exist_ok=True)
    caminho = DADOS / f"{nome}.csv"
    log = DADOS / f"{nome}_consultas.log"
    consultas_feitas, calibracoes_vistas = _ja_coletados(caminho, log)
    agora = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0,
                                                     microsecond=0)
    passos = int(dias * 24 / passo_horas)
    instantes = [agora - dt.timedelta(hours=passo_horas * k)
                 for k in range(passos + 1)][::-1]
    novo = not caminho.exists()
    gravadas = 0
    with caminho.open("a", newline="") as fh, log.open("a") as fh_log:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        if novo:
            escritor.writeheader()
        for i, instante in enumerate(instantes, 1):
            if instante.isoformat() in consultas_feitas:
                continue
            try:
                linhas = consultar(backend, instante, nome)
            except Exception as err:
                print(f"  {instante:%Y-%m-%d %H:%M} falhou: {err}")
                time.sleep(pausa_s)
                continue
            cal = linhas[0]["calibracao_utc"] if linhas else ""
            if cal and cal in calibracoes_vistas:
                status = "calibracao repetida"
            else:
                escritor.writerows(linhas)
                fh.flush()
                calibracoes_vistas.add(cal)
                gravadas += 1
                status = f"calibracao {cal or 'sem data'}"
            fh_log.write(instante.isoformat() + "\n")
            fh_log.flush()
            print(f"  [{i}/{len(instantes)}] {instante:%Y-%m-%d %H:%M} -> {status}")
            time.sleep(pausa_s)
    print(f"{gravadas} snapshots novos gravados em {caminho}")
    return caminho


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", required=True, help="ex.: ibm_torino")
    ap.add_argument("--dias", type=int, default=90)
    ap.add_argument("--passo-horas", type=float, default=24.0)
    ap.add_argument("--canal", default=None,
                    help="ibm_quantum_platform (padrao do Qiskit) ou outro")
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs = {}
    if os.environ.get("QISKIT_IBM_TOKEN"):
        kwargs["token"] = os.environ["QISKIT_IBM_TOKEN"]
        if os.environ.get("QISKIT_IBM_INSTANCE"):
            kwargs["instance"] = os.environ["QISKIT_IBM_INSTANCE"]
    if args.canal:
        kwargs["channel"] = args.canal
    servico = QiskitRuntimeService(**kwargs)
    backend = servico.backend(args.backend)
    print(f"backend {backend.name}: {backend.num_qubits} qubits")
    print(f"mapa de acoplamento em {salvar_mapa(backend, backend.name)}")
    coletar(backend, backend.name, args.dias, args.passo_horas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
