"""QSim Admin Gateway (FastAPI, porta 10000).

Unica fronteira HTTP do sistema. Internamente tudo e NATS + MessagePack;
resultados chegam ao navegador por WebSocket.

Recursos de pesquisa:
  - Submissao por JSON ou OpenQASM (campo qasm);
  - Prioridade (0-9) roteada pelo scheduler;
  - Cancelamento de jobs (fila ou execucao);
  - Historico persistente em SQLite (sobrevive a reinicios);
  - Exportacao de resultados: JSON, CSV e tabela LaTeX para artigos.

Seguranca: Bearer token, rate limit por IP, limite de corpo, validacao
em duas camadas, cabecalhos restritivos, metricas de rejeicao.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, start_http_server
from pydantic import BaseModel, Field, field_validator, model_validator

from shared.bus import Bus
from shared.logging_setup import setup_logging
from shared.settings import settings
from shared.validation import MAX_OPERATIONS, validate_circuit

from services.engine import qasm as qasm_parser

logger = setup_logging("qsim-gateway", settings.log_dir, settings.log_level)

STATIC_DIR = Path(__file__).parent / "static"

JOBS_SUBMITTED = Counter("qsim_gateway_jobs_submitted_total", "Jobs submetidos")
REQUESTS_REJECTED = Counter(
    "qsim_gateway_rejected_total", "Requisicoes rejeitadas", ["reason"]
)

bus = Bus(settings.nats_url, name="qsim-gateway")
jobs: dict[str, dict[str, Any]] = {}
ws_clients: set[WebSocket] = set()
_rate: dict[str, deque] = {}


# ---------------------------------------------------------------------------
# Persistencia (SQLite): historico de jobs e resultados
# ---------------------------------------------------------------------------
class JobStore:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                 job_id TEXT PRIMARY KEY,
                 label TEXT,
                 status TEXT,
                 submitted_at REAL,
                 finished_at REAL,
                 circuit TEXT,
                 result TEXT
               )"""
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_time ON jobs(submitted_at)"
        )
        self.conn.commit()

    def save(self, job: dict, circuit: dict | None = None) -> None:
        with self.lock:
            self.conn.execute(
                """INSERT INTO jobs
                   (job_id, label, status, submitted_at, finished_at,
                    circuit, result)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                     status=excluded.status,
                     finished_at=excluded.finished_at,
                     result=excluded.result""",
                (
                    job["job_id"],
                    job.get("label"),
                    job.get("status"),
                    job.get("submitted_at"),
                    job.get("finished_at"),
                    json.dumps(circuit) if circuit is not None else None,
                    json.dumps(job.get("result"))
                    if job.get("result") is not None else None,
                ),
            )
            self.conn.commit()

    def fetch(self, job_id: str) -> dict | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT job_id, label, status, submitted_at, finished_at, "
                "circuit, result FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def history(self, limit: int = 200) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT job_id, label, status, submitted_at, finished_at, "
                "circuit, result FROM jobs ORDER BY submitted_at DESC LIMIT ?",
                (min(limit, 1000),),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    @staticmethod
    def _row_to_job(row) -> dict:
        job = {
            "job_id": row[0], "label": row[1], "status": row[2],
            "submitted_at": row[3], "finished_at": row[4],
        }
        if row[5]:
            circuit = json.loads(row[5])
            job["num_qubits"] = circuit.get("num_qubits")
            job["shots"] = circuit.get("shots")
            job["circuit"] = circuit
        if row[6]:
            job["result"] = json.loads(row[6])
        return job


store = JobStore(settings.db_path)


# ---------------------------------------------------------------------------
# Modelos de entrada
# ---------------------------------------------------------------------------
class Operation(BaseModel):
    model_config = {"extra": "forbid"}
    op: Literal["gate", "measure", "reset", "barrier"] = "gate"
    gate: str | None = Field(default=None, max_length=16)
    targets: list[int] = Field(default_factory=list, max_length=3)
    params: list[float] | None = Field(default=None, max_length=8)


class CircuitRequest(BaseModel):
    model_config = {"extra": "forbid"}
    num_qubits: int | None = Field(default=None, ge=1, le=31)
    shots: int = Field(default=1024, ge=1, le=1_000_000)
    seed: int | None = None
    precision: Literal["complex64", "complex128"] = "complex128"
    backend: Literal["numpy", "cupy", "custatevec", "aer", "aer-gpu", "auto"] = "auto"
    fuse: bool = True
    verify: bool = False
    parallel: bool = True
    return_statevector: bool = False
    return_bloch: bool = False
    return_entanglement: bool = False
    observables: list[str] | None = Field(default=None, max_length=31)
    noise: dict[str, Any] | None = None
    operations: list[Operation] | None = Field(
        default=None, max_length=MAX_OPERATIONS
    )
    qasm: str | None = Field(default=None, max_length=200_000)
    label: str | None = Field(default=None, max_length=128)
    priority: int = Field(default=5, ge=0, le=9)

    @field_validator("label")
    @classmethod
    def label_safe(cls, v: str | None) -> str | None:
        if v is not None and not all(c.isalnum() or c in " -_." for c in v):
            raise ValueError("label contem caracteres nao permitidos")
        return v

    @model_validator(mode="after")
    def json_or_qasm(self) -> "CircuitRequest":
        if self.qasm is None and not self.operations:
            raise ValueError("Forneca operations (JSON) ou qasm")
        if self.qasm is None and self.num_qubits is None:
            raise ValueError("num_qubits e obrigatorio com operations")
        return self

    def to_circuit(self) -> dict:
        """Monta o circuito final (resolve QASM se presente)."""
        if self.qasm is not None:
            parsed = qasm_parser.parse(self.qasm)
            num_qubits = parsed["num_qubits"]
            operations = parsed["operations"]
        else:
            num_qubits = self.num_qubits
            operations = [
                op.model_dump(exclude_none=True) for op in self.operations
            ]
        circuit: dict[str, Any] = {
            "num_qubits": num_qubits,
            "shots": self.shots,
            "precision": self.precision,
            "backend": self.backend,
            "fuse": self.fuse,
            "verify": self.verify,
            "parallel": self.parallel,
            "return_statevector": self.return_statevector,
            "return_bloch": self.return_bloch,
            "return_entanglement": self.return_entanglement,
            "operations": operations,
        }
        if self.seed is not None:
            circuit["seed"] = self.seed
        if self.observables:
            circuit["observables"] = self.observables
        if self.noise:
            circuit["noise"] = self.noise
        return circuit


# ---------------------------------------------------------------------------
# Autenticacao e rate limiting
# ---------------------------------------------------------------------------
def check_token(token: str | None) -> bool:
    if not settings.admin_token:
        return True
    return token == settings.admin_token


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.admin_token:
        return
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
    if not check_token(token):
        REQUESTS_REJECTED.labels(reason="auth").inc()
        raise HTTPException(status_code=401, detail="Token invalido ou ausente")


def rate_limited(client_ip: str) -> bool:
    now = time.monotonic()
    window = _rate.setdefault(client_ip, deque())
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        return True
    window.append(now)
    return False


# ---------------------------------------------------------------------------
# Ciclo de vida
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.admin_token:
        logger.warning(
            "QSIM_ADMIN_TOKEN nao definido: gateway SEM autenticacao "
            "(aceitavel apenas em desenvolvimento local)"
        )
    await bus.connect()
    logger.info("Gateway conectado ao NATS em %s", settings.nats_url)
    await bus.subscribe(f"{settings.subject_results}.*", on_result)
    await bus.subscribe(f"{settings.subject_telemetry}.*", on_telemetry)
    start_http_server(settings.metrics_port_gateway)
    logger.info("Metricas Prometheus em :%d/metrics",
                settings.metrics_port_gateway)
    yield
    await bus.close()


app = FastAPI(title="QSim Admin Gateway", version="0.3.0", lifespan=lifespan)


@app.middleware("http")
async def guard_middleware(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and int(length) > settings.max_body_bytes:
        REQUESTS_REJECTED.labels(reason="body_size").inc()
        return JSONResponse(status_code=413,
                            content={"detail": "Payload muito grande"})
    if request.url.path.startswith("/api"):
        ip = request.client.host if request.client else "unknown"
        if rate_limited(ip):
            REQUESTS_REJECTED.labels(reason="rate_limit").inc()
            return JSONResponse(status_code=429,
                                content={"detail": "Limite de requisicoes excedido"})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:"
    )
    return response


# ---------------------------------------------------------------------------
# Handlers NATS -> WebSocket
# ---------------------------------------------------------------------------
async def on_result(subject: str, payload: dict) -> None:
    job_id = payload.get("job_id")
    if job_id in jobs:
        jobs[job_id].update(payload)
        jobs[job_id]["finished_at"] = time.time()
        store.save(jobs[job_id])
    logger.info("Resultado: job %s status %s", job_id, payload.get("status"))
    await broadcast({"type": "job_update", "data": jobs.get(job_id, payload)})


async def on_telemetry(subject: str, payload: dict) -> None:
    await broadcast({"type": "telemetry", "subject": subject, "data": payload})


async def broadcast(message: dict) -> None:
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# API REST
# ---------------------------------------------------------------------------
@app.post("/api/jobs", dependencies=[Depends(require_auth)])
async def submit_job(req: CircuitRequest, request: Request) -> dict:
    try:
        circuit = req.to_circuit()
        validate_circuit(circuit)
    except ValueError as err:
        REQUESTS_REJECTED.labels(reason="validation").inc()
        raise HTTPException(status_code=422, detail=str(err))

    job_id = uuid.uuid4().hex[:12]
    origin = request.client.host if request.client else "unknown"
    job = {
        "job_id": job_id,
        "status": "queued",
        "label": req.label or f"job-{job_id}",
        "num_qubits": circuit["num_qubits"],
        "shots": circuit["shots"],
        "precision": circuit["precision"],
        "verify": circuit["verify"],
        "priority": req.priority,
        "submitted_at": time.time(),
    }
    jobs[job_id] = job
    store.save(job, circuit=circuit)
    await bus.publish(settings.subject_submit, {
        "job_id": job_id, "circuit": circuit,
        "origin": origin, "priority": req.priority,
    })
    JOBS_SUBMITTED.inc()
    logger.info("Job %s submetido (%d qubits, %d shots, %s, prio %d)",
                job_id, circuit["num_qubits"], circuit["shots"],
                circuit["precision"], req.priority)
    await broadcast({"type": "job_update", "data": job})
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
async def cancel_job(job_id: str) -> dict:
    if job_id not in jobs and store.fetch(job_id) is None:
        raise HTTPException(status_code=404, detail="Job nao encontrado")
    await bus.publish(f"{settings.subject_control}.cancel.{job_id}",
                      {"job_id": job_id})
    logger.info("Cancelamento solicitado para job %s", job_id)
    return {"job_id": job_id, "status": "cancel_requested"}


@app.get("/api/jobs", dependencies=[Depends(require_auth)])
async def list_jobs() -> list[dict]:
    return sorted(jobs.values(), key=lambda j: j["submitted_at"], reverse=True)


@app.get("/api/history", dependencies=[Depends(require_auth)])
async def history(limit: int = 200) -> list[dict]:
    return store.history(limit)


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
async def get_job(job_id: str) -> dict:
    if job_id in jobs:
        return jobs[job_id]
    stored = store.fetch(job_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Job nao encontrado")
    return stored


@app.get("/api/jobs/{job_id}/export", dependencies=[Depends(require_auth)])
async def export_job(job_id: str, fmt: str = "json"):
    job = jobs.get(job_id) or store.fetch(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nao encontrado")
    result = job.get("result")
    if not result:
        raise HTTPException(status_code=409, detail="Job ainda sem resultado")

    if fmt == "json":
        return JSONResponse(
            content={"job": {k: v for k, v in job.items() if k != "circuit"},
                     "circuit": job.get("circuit")},
            headers={"Content-Disposition":
                     f"attachment; filename=qsim-{job_id}.json"},
        )
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["state", "counts", "probability"])
        shots = result.get("shots", 1)
        for state, count in sorted(result.get("counts", {}).items()):
            writer.writerow([state, count, count / shots])
        return PlainTextResponse(
            buf.getvalue(), media_type="text/csv",
            headers={"Content-Disposition":
                     f"attachment; filename=qsim-{job_id}.csv"},
        )
    if fmt == "latex":
        return PlainTextResponse(
            _to_latex(job, result), media_type="text/plain",
            headers={"Content-Disposition":
                     f"attachment; filename=qsim-{job_id}.tex"},
        )
    raise HTTPException(status_code=422, detail="fmt deve ser json, csv ou latex")


def _to_latex(job: dict, result: dict) -> str:
    shots = result.get("shots", 1)
    rows = "\n".join(
        f"    $\\ket{{{state}}}$ & {count} & {count / shots:.4f} \\\\"
        for state, count in sorted(result.get("counts", {}).items())[:64]
    )
    meta = result.get("metadata", {})
    caption = (
        f"Resultados QSim: {job.get('label', job['job_id'])} "
        f"({result.get('num_qubits')} qubits, {shots} shots, "
        f"{meta.get('precision', '?')}, modo {meta.get('mode', '?')})"
    )
    return (
        "% Gerado pelo QSim - requer \\usepackage{braket} e \\usepackage{booktabs}\n"
        "\\begin{table}[htbp]\n  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        "  \\begin{tabular}{lrr}\n    \\toprule\n"
        "    Estado & Contagens & Probabilidade \\\\\n    \\midrule\n"
        f"{rows}\n    \\bottomrule\n  \\end{{tabular}}\n\\end{{table}}\n"
    )


@app.get("/api/engines", dependencies=[Depends(require_auth)])
async def list_engines() -> dict:
    try:
        info = await bus.request(
            f"{settings.subject_control}.engine.info", {}, timeout=2.0
        )
        return {"engines": [info]}
    except Exception:
        return {"engines": []}


@app.get("/api/scheduler", dependencies=[Depends(require_auth)])
async def scheduler_stats() -> dict:
    try:
        return await bus.request(
            f"{settings.subject_control}.scheduler.stats", {}, timeout=2.0
        )
    except Exception:
        return {"error": "scheduler indisponivel"}


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "nats": bus.nc is not None and bus.nc.is_connected,
        "jobs_tracked": len(jobs),
        "auth_enabled": bool(settings.admin_token),
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    if not check_token(ws.query_params.get("token")):
        await ws.close(code=4401)
        return
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Console web
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.gateway.main:app",
        host=settings.admin_host,
        port=settings.admin_port,
        log_level=settings.log_level.lower(),
    )
