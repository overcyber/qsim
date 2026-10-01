"""QSim Scheduler.

Recebe submissoes em qsim.jobs.submit e despacha para a fila da menor
classe que comporta o job:

  small    n <= 20                 (todos os workers)
  medium   21 <= n <= 29           (workers com capacidade media/GPU)
  large    n >= 30                 (GPUs de 24 GB)
  noisy    ruido habilitado        (trajetorias; CPU paralela + GPUs ociosas)

Funcoes:
  - Fila de prioridades (0-9; maior executa primeiro sob backlog);
  - Quotas por origem (jobs ativos simultaneos, resposta rejected acima);
  - Cancelamento: remove da fila se ainda nao despachado e repassa o
    cancelamento aos workers se ja estiver em execucao;
  - Estatisticas via request-reply em qsim.control.scheduler.stats.

Zero HTTP: apenas NATS + MessagePack.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import signal
import time

from prometheus_client import Counter, Gauge, start_http_server

from shared.bus import Bus
from shared.logging_setup import setup_logging
from shared.settings import settings

logger = setup_logging("qsim-scheduler", settings.log_dir, settings.log_level)

DISPATCHED = Counter("qsim_scheduler_dispatched_total", "Jobs despachados",
                     ["job_class"])
REJECTED = Counter("qsim_scheduler_rejected_total", "Jobs rejeitados",
                   ["reason"])
QUEUE_DEPTH = Gauge("qsim_scheduler_queue_depth", "Jobs aguardando despacho")
ACTIVE = Gauge("qsim_scheduler_active_jobs", "Jobs despachados sem resultado")


def classify(circuit: dict) -> str:
    noise = circuit.get("noise") or {}
    if any(v for v in noise.values() if v):
        return "noisy"
    n = int(circuit.get("num_qubits", 1))
    if n >= settings.class_large_min:
        return "large"
    if n >= settings.class_medium_min:
        return "medium"
    return "small"


class Scheduler:
    def __init__(self) -> None:
        self.bus = Bus(settings.nats_url, name="qsim-scheduler")
        self.heap: list[tuple[int, int, dict]] = []
        self.counter = itertools.count()
        self.wakeup = asyncio.Event()
        self.cancelled: set[str] = set()
        self.active_by_origin: dict[str, int] = {}
        self.origin_of: dict[str, str] = {}
        self.started_at = time.time()
        self.running = True

    async def start(self) -> None:
        await self.bus.connect()
        logger.info("Scheduler conectado ao NATS em %s", settings.nats_url)

        if settings.use_jetstream:
            ok = await self.bus.ensure_workqueue(
                settings.nats_stream, [f"{settings.subject_jobs}.>"]
            )
            logger.info("JetStream work queue: %s",
                        "ativa" if ok else "indisponivel (fallback core NATS)")

        await self.bus.subscribe(settings.subject_submit, self.on_submit)
        await self.bus.subscribe(f"{settings.subject_results}.*", self.on_result)
        await self.bus.subscribe(f"{settings.subject_control}.cancel.*",
                                 self.on_cancel)
        await self.bus.respond_to(f"{settings.subject_control}.scheduler.stats",
                                  self.stats)
        await self.dispatch_loop()

    # ------------------------------------------------------------------
    async def on_submit(self, subject: str, envelope: dict) -> None:
        job_id = str(envelope.get("job_id", ""))
        circuit = envelope.get("circuit")
        origin = str(envelope.get("origin", "unknown"))
        priority = int(envelope.get("priority", 5))
        priority = max(0, min(9, priority))

        if not job_id or not isinstance(circuit, dict):
            REJECTED.labels(reason="malformed").inc()
            return

        active = self.active_by_origin.get(origin, 0)
        if active >= settings.max_active_per_origin:
            REJECTED.labels(reason="quota").inc()
            logger.warning("Job %s rejeitado: quota da origem %s (%d ativos)",
                           job_id, origin, active)
            await self.bus.publish(
                f"{settings.subject_results}.{job_id}",
                {"job_id": job_id, "status": "rejected",
                 "error": f"Quota excedida: {active} jobs ativos da origem"},
            )
            return

        heapq.heappush(self.heap, (-priority, next(self.counter), envelope))
        QUEUE_DEPTH.set(len(self.heap))
        self.wakeup.set()

    async def dispatch_loop(self) -> None:
        while self.running:
            if not self.heap:
                self.wakeup.clear()
                try:
                    await asyncio.wait_for(self.wakeup.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
            if not self.heap:
                continue

            _, _, envelope = heapq.heappop(self.heap)
            QUEUE_DEPTH.set(len(self.heap))
            job_id = envelope["job_id"]
            origin = str(envelope.get("origin", "unknown"))

            if job_id in self.cancelled:
                self.cancelled.discard(job_id)
                await self.bus.publish(
                    f"{settings.subject_results}.{job_id}",
                    {"job_id": job_id, "status": "cancelled",
                     "error": "Cancelado antes do despacho"},
                )
                continue

            job_class = classify(envelope["circuit"])
            self.active_by_origin[origin] = self.active_by_origin.get(origin, 0) + 1
            self.origin_of[job_id] = origin
            ACTIVE.set(sum(self.active_by_origin.values()))
            DISPATCHED.labels(job_class=job_class).inc()

            await self.bus.publish(
                f"{settings.subject_jobs}.{job_class}",
                {"job_id": job_id, "circuit": envelope["circuit"],
                 "job_class": job_class},
            )
            logger.info("Job %s despachado para classe %s", job_id, job_class)

    # ------------------------------------------------------------------
    async def on_result(self, subject: str, payload: dict) -> None:
        job_id = str(payload.get("job_id", ""))
        origin = self.origin_of.pop(job_id, None)
        if origin:
            self.active_by_origin[origin] = max(
                0, self.active_by_origin.get(origin, 1) - 1
            )
            if self.active_by_origin[origin] == 0:
                self.active_by_origin.pop(origin, None)
            ACTIVE.set(sum(self.active_by_origin.values()))
        self.cancelled.discard(job_id)

    async def on_cancel(self, subject: str, payload: dict) -> None:
        job_id = subject.rsplit(".", 1)[-1]
        self.cancelled.add(job_id)
        logger.info("Cancelamento registrado para job %s", job_id)

    async def stats(self, _data: dict) -> dict:
        return {
            "queue_depth": len(self.heap),
            "active_jobs": sum(self.active_by_origin.values()),
            "origins": dict(self.active_by_origin),
            "uptime_s": time.time() - self.started_at,
        }

    def stop(self) -> None:
        self.running = False
        self.wakeup.set()


async def main() -> None:
    start_http_server(settings.metrics_port_scheduler)
    logger.info("Metricas Prometheus em :%d/metrics",
                settings.metrics_port_scheduler)
    sched = Scheduler()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, sched.stop)
    await sched.start()


if __name__ == "__main__":
    asyncio.run(main())
