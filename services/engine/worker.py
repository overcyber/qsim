"""QSim Engine Worker.

Consome jobs das filas de classe que sua capacidade permite:

  small   sempre
  noisy   sempre (trajetorias paralelas nos nucleos da CPU)
  medium  se backend GPU disponivel (ou liberado via QSIM_ENGINE_CLASSES)
  large   se backend GPU com VRAM suficiente

Filas via JetStream work queue (duraveis; jobs de um worker morto sao
reentregues) com fallback transparente para queue groups core NATS.

Seguranca: o worker NUNCA confia no payload da fila — revalida todo
circuito com shared.validation antes de executar. Cancelamento cooperativo
via qsim.control.cancel.<job_id>.

Metricas Prometheus em :9101/metrics.
"""
from __future__ import annotations

import asyncio
import os
import platform
import signal
import threading
import time
import traceback
import uuid

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from shared.bus import Bus
from shared.logging_setup import setup_logging
from shared.settings import settings
from shared.validation import validate_circuit

from services.engine.backends import available_backends
from services.engine.simulator import MAX_QUBITS, JobCancelled, run_circuit

logger = setup_logging("qsim-engine", settings.log_dir, settings.log_level)

WORKER_ID = f"engine-{uuid.uuid4().hex[:8]}"

JOBS_TOTAL = Counter("qsim_engine_jobs_total", "Jobs processados", ["status"])
JOB_DURATION = Histogram(
    "qsim_engine_job_duration_seconds", "Duracao dos jobs",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 15, 60, 180),
)
ACTIVE_JOBS = Gauge("qsim_engine_active_jobs", "Jobs em execucao")
QUBITS_HIST = Histogram(
    "qsim_engine_qubits", "Qubits por job",
    buckets=tuple(range(1, MAX_QUBITS + 1)),
)


def my_classes() -> list[str]:
    """Classes de job que este worker atende."""
    configured = settings.engine_classes.strip().lower()
    if configured != "auto":
        return [c.strip() for c in configured.split(",") if c.strip()]
    backends = available_backends()
    classes = ["small", "noisy"]
    if "cupy" in backends:
        classes += ["medium", "large"]
    elif "aer" in backends:
        classes.append("medium")  # Aer C++ em CPU: 21-29 qubits com RAM suficiente
    return classes


class EngineWorker:
    def __init__(self) -> None:
        self.bus = Bus(settings.nats_url, name=WORKER_ID)
        self.running = True
        self.started_at = time.time()
        self.cancel_flags: dict[str, threading.Event] = {}

    async def start(self) -> None:
        await self.bus.connect()
        classes = my_classes()
        logger.info("Worker %s conectado; classes: %s; backends: %s",
                    WORKER_ID, classes, available_backends())

        if os.environ.get("QSIM_USE_NUMBA", "0") == "1":
            try:
                from services.engine import numba_kernels
                numba_kernels.warmup()
                logger.info("Kernels Numba compilados (aceleracao 1q ativa)")
            except Exception as err:
                logger.warning("Numba solicitado mas indisponivel: %s", err)

        if settings.use_jetstream:
            await self.bus.ensure_workqueue(
                settings.nats_stream, [f"{settings.subject_jobs}.>"]
            )

        for cls in classes:
            subject = f"{settings.subject_jobs}.{cls}"
            if settings.use_jetstream:
                await self.bus.subscribe_workqueue(
                    subject, self.handle_job,
                    queue=f"{settings.engine_queue_group}-{cls}",
                    durable=f"{settings.engine_queue_group}-{cls}",
                    stream=settings.nats_stream,
                )
            else:
                await self.bus.subscribe(
                    subject, self.handle_job,
                    queue=f"{settings.engine_queue_group}-{cls}",
                )

        # Compatibilidade: submissao direta sem scheduler (dev)
        await self.bus.subscribe(settings.subject_jobs, self.handle_job,
                                 queue=settings.engine_queue_group)

        await self.bus.subscribe(f"{settings.subject_control}.cancel.*",
                                 self.on_cancel)
        await self.bus.respond_to(f"{settings.subject_control}.engine.ping",
                                  self.ping)
        await self.bus.respond_to(f"{settings.subject_control}.engine.info",
                                  self.info)
        await self.telemetry_loop()

    # ------------------------------------------------------------------
    async def on_cancel(self, subject: str, _payload: dict) -> None:
        job_id = subject.rsplit(".", 1)[-1]
        flag = self.cancel_flags.get(job_id)
        if flag:
            flag.set()
            logger.info("Cancelamento solicitado para job %s em execucao",
                        job_id)

    async def handle_job(self, subject: str, job: dict) -> None:
        job_id = str(job.get("job_id", "sem-id"))
        circuit = job.get("circuit", {})
        result_subject = f"{settings.subject_results}.{job_id}"

        cancel_flag = threading.Event()
        self.cancel_flags[job_id] = cancel_flag

        ACTIVE_JOBS.inc()
        t0 = time.perf_counter()
        logger.info(
            "Job %s recebido em %s: %s qubits, %s shots",
            job_id, subject, circuit.get("num_qubits"), circuit.get("shots"),
        )
        try:
            # O worker NUNCA confia na fila: revalida antes de executar.
            validate_circuit(circuit)
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: run_circuit(
                        circuit, should_cancel=cancel_flag.is_set
                    )
                ),
                timeout=settings.job_timeout_s,
            )
            payload = {
                "job_id": job_id,
                "status": "done",
                "worker": WORKER_ID,
                "result": result.to_dict(),
            }
            JOBS_TOTAL.labels(status="done").inc()
            QUBITS_HIST.observe(int(circuit.get("num_qubits", 0)))
            logger.info("Job %s concluido em %.1f ms", job_id,
                        result.elapsed_ms)
        except ValueError as err:
            payload = {"job_id": job_id, "status": "rejected",
                       "worker": WORKER_ID, "error": str(err)}
            JOBS_TOTAL.labels(status="rejected").inc()
            logger.warning("Job %s rejeitado na validacao: %s", job_id, err)
        except JobCancelled:
            payload = {"job_id": job_id, "status": "cancelled",
                       "worker": WORKER_ID,
                       "error": "Cancelado durante a execucao"}
            JOBS_TOTAL.labels(status="cancelled").inc()
            logger.info("Job %s cancelado", job_id)
        except asyncio.TimeoutError:
            payload = {"job_id": job_id, "status": "timeout",
                       "worker": WORKER_ID,
                       "error": f"Job excedeu {settings.job_timeout_s}s"}
            JOBS_TOTAL.labels(status="timeout").inc()
            logger.error("Job %s: timeout", job_id)
        except Exception as err:
            payload = {"job_id": job_id, "status": "error",
                       "worker": WORKER_ID, "error": str(err)}
            JOBS_TOTAL.labels(status="error").inc()
            logger.error("Job %s falhou: %s\n%s", job_id, err,
                         traceback.format_exc())
        finally:
            ACTIVE_JOBS.dec()
            JOB_DURATION.observe(time.perf_counter() - t0)
            self.cancel_flags.pop(job_id, None)

        await self.bus.publish(result_subject, payload)

    # ------------------------------------------------------------------
    async def ping(self, _data: dict) -> dict:
        return {"worker": WORKER_ID, "status": "ok",
                "uptime_s": time.time() - self.started_at}

    async def info(self, _data: dict) -> dict:
        return {
            "worker": WORKER_ID,
            "classes": my_classes(),
            "max_qubits": MAX_QUBITS,
            "max_shots": settings.max_shots,
            "backends": available_backends(),
            "precisions": ["complex64", "complex128"],
            "trajectory_workers": settings.trajectory_workers or os.cpu_count(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        }

    # ------------------------------------------------------------------
    async def telemetry_loop(self) -> None:
        while self.running:
            await self.bus.publish(
                f"{settings.subject_telemetry}.engine",
                {
                    "worker": WORKER_ID,
                    "ts": time.time(),
                    "classes": my_classes(),
                    "active_jobs": ACTIVE_JOBS._value.get(),
                    "uptime_s": time.time() - self.started_at,
                },
            )
            await asyncio.sleep(5)

    def stop(self) -> None:
        self.running = False


async def main() -> None:
    start_http_server(settings.metrics_port_engine)
    logger.info("Metricas Prometheus em :%d/metrics",
                settings.metrics_port_engine)
    worker = EngineWorker()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.stop)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
