"""Manifesto de proveniencia anexado a todo resultado do experimento.

Registra o suficiente para um terceiro reproduzir ou auditar o numero:
commit e estado do repositorio, versoes das bibliotecas que tocam a
fisica ou a estatistica, plataforma e o hash SHA-256 de cada arquivo-fonte
do experimento e do motor usado.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

ARQUIVOS_RASTREADOS = [
    "experiments/tema1_canario/modelo.py",
    "experiments/tema1_canario/detectores.py",
    "experiments/tema1_canario/cenarios.py",
    "experiments/tema1_canario/piloto_binomial.py",
    "experiments/tema1_canario/benchmark_sequencial.py",
    "services/engine/simulator.py",
    "services/engine/backend_aer.py",
    "services/engine/gates.py",
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _versao(modulo: str) -> str:
    try:
        mod = __import__(modulo)
        return str(getattr(mod, "__version__", "desconhecida"))
    except Exception:
        return "ausente"


def _sha256(caminho: Path) -> str:
    try:
        return hashlib.sha256(caminho.read_bytes()).hexdigest()
    except OSError:
        return "ausente"


def manifesto(extra: dict | None = None) -> dict:
    # Arquivos nao rastreados (ex.: resultados recem-gerados) nao alteram o
    # codigo; so modificacoes em arquivos versionados sujam a arvore.
    status = _git("status", "--porcelain", "--untracked-files=no")
    m = {
        "gerado_em_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "arvore_suja": bool(status),
        },
        "versoes": {
            "python": platform.python_version(),
            "numpy": _versao("numpy"),
            "qiskit": _versao("qiskit"),
            "qiskit_aer": _versao("qiskit_aer"),
        },
        "plataforma": f"{platform.system()} {platform.machine()}",
        "sha256": {arq: _sha256(RAIZ / arq) for arq in ARQUIVOS_RASTREADOS},
    }
    if extra:
        m.update(extra)
    return m
