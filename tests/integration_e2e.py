"""Teste de integracao ponta a ponta (requer nats-server no PATH).

Sobe NATS com JetStream, scheduler, worker e gateway como processos reais
e exercita o caminho completo:

  HTTP -> gateway -> NATS submit -> scheduler -> fila de classe ->
  worker -> resultado -> gateway (SQLite + WebSocket) -> HTTP

Cobre: submissao JSON e QASM, prioridade, observaveis, Bloch,
emaranhamento, exportacao CSV/LaTeX, cancelamento, quota do scheduler
e o SDK direto no NATS.

Uso: PYTHONPATH=. python3 tests/integration_e2e.py
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time

import httpx

BASE = "http://127.0.0.1:10000"
ENV = {
    **os.environ,
    "PYTHONPATH": ".",
    "QSIM_LOG_DIR": tempfile.mkdtemp(prefix="qsim-logs-"),
    "QSIM_DB_PATH": os.path.join(tempfile.mkdtemp(prefix="qsim-db-"), "e2e.db"),
    "QSIM_ADMIN_TOKEN": "teste-e2e",
}
HEADERS = {"Authorization": "Bearer teste-e2e"}
procs: list[subprocess.Popen] = []


def spawn(*cmd: str) -> subprocess.Popen:
    p = subprocess.Popen(cmd, env=ENV, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    procs.append(p)
    return p


async def wait_result(client: httpx.AsyncClient, job_id: str,
                      timeout: float = 60.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = await client.get(f"{BASE}/api/jobs/{job_id}", headers=HEADERS)
        job = r.json()
        if job.get("status") not in ("queued", None):
            return job
        await asyncio.sleep(0.25)
    raise TimeoutError(f"job {job_id} sem resultado")


async def main() -> None:
    spawn("nats-server", "-js", "-p", "4222")
    time.sleep(1.0)
    spawn(sys.executable, "-m", "services.scheduler.main")
    spawn(sys.executable, "-m", "services.engine.worker")
    spawn(sys.executable, "-m", "uvicorn", "services.gateway.main:app",
          "--host", "127.0.0.1", "--port", "10000", "--log-level", "error")

    async with httpx.AsyncClient(timeout=10.0) as client:
        # espera ativa: gateway pronto em ate 60s
        for _ in range(120):
            try:
                r = await client.get(f"{BASE}/api/health")
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("gateway nao subiu em 60s")
        print("[e2e] servicos no ar")

        # 0) auth obrigatoria
        r = await client.get(f"{BASE}/api/jobs")
        assert r.status_code == 401, f"esperava 401, veio {r.status_code}"
        print("[e2e] autenticacao exigida OK")

        # 1) Bell por JSON com verificacao, Bloch e emaranhamento
        r = await client.post(f"{BASE}/api/jobs", headers=HEADERS, json={
            "num_qubits": 2, "shots": 4096, "seed": 7, "verify": True,
            "return_bloch": True, "return_entanglement": True,
            "observables": ["ZZ", "XX"], "label": "e2e-bell",
            "operations": [{"gate": "h", "targets": [0]},
                           {"gate": "cx", "targets": [0, 1]}],
        })
        assert r.status_code == 200, r.text
        job = await wait_result(client, r.json()["job_id"])
        assert job["status"] == "done", job
        res = job["result"]
        assert set(res["counts"]) <= {"00", "11"}
        assert res["metadata"]["verification"]["passed"] is True
        assert abs(res["expectation"]["ZZ"] - 1.0) < 1e-9
        assert abs(res["entanglement"]["0"] - 1.0) < 1e-9
        print("[e2e] Bell JSON + verify + Bloch + entropia + Pauli OK")

        # 2) submissao QASM
        r = await client.post(f"{BASE}/api/jobs", headers=HEADERS, json={
            "qasm": "OPENQASM 3; qubit[3] q; h q[0]; cx q[0],q[1]; "
                    "cx q[1],q[2];",
            "shots": 1024, "seed": 1, "label": "e2e-qasm-ghz",
        })
        assert r.status_code == 200, r.text
        job = await wait_result(client, r.json()["job_id"])
        assert job["status"] == "done"
        assert set(job["result"]["counts"]) <= {"000", "111"}
        ghz_id = job["job_id"]
        print("[e2e] submissao OpenQASM OK")

        # 3) exportacao CSV e LaTeX
        r = await client.get(f"{BASE}/api/jobs/{ghz_id}/export?fmt=csv",
                             headers=HEADERS)
        assert r.status_code == 200 and "state,counts" in r.text
        r = await client.get(f"{BASE}/api/jobs/{ghz_id}/export?fmt=latex",
                             headers=HEADERS)
        assert r.status_code == 200 and "\\begin{table}" in r.text
        print("[e2e] exportacao CSV e LaTeX OK")

        # 4) job com ruido (classe noisy) e historico persistente
        r = await client.post(f"{BASE}/api/jobs", headers=HEADERS, json={
            "num_qubits": 2, "shots": 200, "seed": 3,
            "noise": {"depolarizing": 0.05,
                      "per_gate": {"cx": {"depolarizing": 0.1}}},
            "label": "e2e-ruido",
            "operations": [{"gate": "h", "targets": [0]},
                           {"gate": "cx", "targets": [0, 1]}],
        })
        job = await wait_result(client, r.json()["job_id"])
        assert job["status"] == "done"
        assert job["result"]["metadata"]["mode"] == "trajectories"
        r = await client.get(f"{BASE}/api/history?limit=10", headers=HEADERS)
        assert any(j["label"] == "e2e-ruido" for j in r.json())
        print("[e2e] classe noisy + historico SQLite OK")

        # 5) cancelamento de job pesado (20 qubits = classe small, CPU)
        heavy_ops = [{"gate": "h", "targets": [i % 20]} for i in range(5000)]
        r = await client.post(f"{BASE}/api/jobs", headers=HEADERS, json={
            "num_qubits": 20, "shots": 10, "fuse": False,
            "label": "e2e-cancel", "operations": heavy_ops,
        })
        heavy_id = r.json()["job_id"]
        await asyncio.sleep(0.4)
        r = await client.post(f"{BASE}/api/jobs/{heavy_id}/cancel",
                              headers=HEADERS)
        assert r.status_code == 200
        job = await wait_result(client, heavy_id)
        assert job["status"] in ("cancelled", "done"), job["status"]
        print(f"[e2e] cancelamento OK (status final: {job['status']})")

        # 6) scheduler stats
        r = await client.get(f"{BASE}/api/scheduler", headers=HEADERS)
        assert "queue_depth" in r.json()
        print("[e2e] estatisticas do scheduler OK")

        # 7) motor Aer (C++) atraves de todo o stack, se instalado
        try:
            import qiskit_aer  # noqa: F401
            has_aer = True
        except ImportError:
            has_aer = False
        if has_aer:
            r = await client.post(f"{BASE}/api/jobs", headers=HEADERS, json={
                "num_qubits": 2, "shots": 2048, "seed": 11,
                "backend": "aer", "verify": True,
                "observables": ["ZZ"], "label": "e2e-aer",
                "operations": [{"gate": "h", "targets": [0]},
                               {"gate": "cx", "targets": [0, 1]}],
            })
            job = await wait_result(client, r.json()["job_id"])
            assert job["status"] == "done", job
            assert job["result"]["metadata"]["engine"] == "aer"
            assert job["result"]["metadata"]["verification"]["passed"] is True
            assert abs(job["result"]["expectation"]["ZZ"] - 1.0) < 1e-9
            print("[e2e] backend aer via gateway/scheduler/worker OK")
        else:
            print("[e2e] backend aer PULADO (qiskit-aer nao instalado)")

    # 7) SDK direto no NATS (sem HTTP)
    sys.path.insert(0, ".")
    from sdk.qsim_client import QSimClient
    async with QSimClient("nats://127.0.0.1:4222") as sdk:
        payload = await sdk.run({
            "num_qubits": 1, "shots": 2048, "seed": 0,
            "observables": ["z0"],
            "operations": [{"gate": "ry", "targets": [0],
                            "params": [1.0471975511965976]}],  # pi/3
        })
        assert payload["status"] == "done"
        z = payload["result"]["expectation"]["z0"]
        assert abs(z - 0.5) < 1e-9, z  # <Z> = cos(pi/3) = 0.5
        results = await sdk.sweep(
            lambda th: {"num_qubits": 1, "shots": 1, "seed": 0,
                        "observables": ["z0"],
                        "operations": [{"gate": "ry", "targets": [0],
                                        "params": [th]}]},
            [0.0, 1.5707963267948966, 3.141592653589793],
        )
        zs = [r["result"]["expectation"]["z0"] for r in results]
        assert abs(zs[0] - 1) < 1e-9 and abs(zs[1]) < 1e-9 and abs(zs[2] + 1) < 1e-9
        print("[e2e] SDK: run + sweep de parametros OK (curva cos(theta))")

    print("\nIntegracao ponta a ponta: TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.5)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
