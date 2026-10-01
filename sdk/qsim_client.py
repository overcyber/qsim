"""SDK Python do QSim para pesquisa.

Fala diretamente com o NATS (sem passar pelo gateway HTTP): submissao,
espera de resultado e varreduras de parametros — o fluxo tipico de
experimentos (curvas de decoerencia, scans de angulo, VQE-like).

Uso assicrono:

    from sdk.qsim_client import QSimClient

    async with QSimClient("nats://127.0.0.1:4222") as client:
        result = await client.run({
            "num_qubits": 2, "shots": 4096,
            "operations": [{"gate": "h", "targets": [0]},
                           {"gate": "cx", "targets": [0, 1]}],
        })
        print(result["result"]["counts"])

Uso sincrono (scripts de pesquisa):

    from sdk.qsim_client import run_sync
    result = run_sync(circuit)

Varredura de parametros:

    def circuito(theta):
        return {"num_qubits": 1, "shots": 2048,
                "observables": ["z0"],
                "operations": [{"gate": "ry", "targets": [0],
                                "params": [theta]}]}

    resultados = sweep_sync(circuito, [i * 0.1 for i in range(63)])
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Iterable

import msgpack
import nats

DEFAULT_URL = "nats://127.0.0.1:4222"
SUBJECT_SUBMIT = "qsim.jobs.submit"
SUBJECT_RESULTS = "qsim.jobs.result"
SUBJECT_CANCEL = "qsim.control.cancel"


class QSimClient:
    """Cliente assicrono do QSim via NATS."""

    def __init__(self, url: str = DEFAULT_URL, origin: str = "sdk"):
        self.url = url
        self.origin = origin
        self.nc = None

    async def __aenter__(self) -> "QSimClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def connect(self) -> None:
        self.nc = await nats.connect(self.url, name=f"qsim-sdk-{self.origin}")

    async def close(self) -> None:
        if self.nc:
            await self.nc.drain()

    # ------------------------------------------------------------------
    async def submit(self, circuit: dict[str, Any],
                     priority: int = 5) -> str:
        """Submete um circuito e retorna o job_id (nao espera)."""
        job_id = uuid.uuid4().hex[:12]
        await self.nc.publish(SUBJECT_SUBMIT, msgpack.packb({
            "job_id": job_id,
            "circuit": circuit,
            "origin": self.origin,
            "priority": int(priority),
        }, use_bin_type=True))
        return job_id

    async def run(self, circuit: dict[str, Any], priority: int = 5,
                  timeout: float = 300.0) -> dict[str, Any]:
        """Submete e espera o resultado (assina o subject antes de publicar
        para nao perder a resposta)."""
        job_id = uuid.uuid4().hex[:12]
        future: asyncio.Future = asyncio.get_running_loop().create_future()

        async def _on_result(msg) -> None:
            if not future.done():
                future.set_result(msgpack.unpackb(msg.data, raw=False))

        sub = await self.nc.subscribe(f"{SUBJECT_RESULTS}.{job_id}",
                                      cb=_on_result)
        try:
            await self.nc.publish(SUBJECT_SUBMIT, msgpack.packb({
                "job_id": job_id,
                "circuit": circuit,
                "origin": self.origin,
                "priority": int(priority),
            }, use_bin_type=True))
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            await sub.unsubscribe()

    async def cancel(self, job_id: str) -> None:
        await self.nc.publish(f"{SUBJECT_CANCEL}.{job_id}",
                              msgpack.packb({"job_id": job_id},
                                            use_bin_type=True))

    # ------------------------------------------------------------------
    async def sweep(self, make_circuit: Callable[[Any], dict],
                    values: Iterable[Any], concurrency: int = 4,
                    timeout: float = 300.0) -> list[dict[str, Any]]:
        """Varredura de parametros: executa make_circuit(v) para cada v,
        com ate `concurrency` jobs simultaneos, preservando a ordem."""
        values = list(values)
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(v: Any) -> dict:
            async with semaphore:
                payload = await self.run(make_circuit(v), timeout=timeout)
                return {"value": v, **payload}

        return list(await asyncio.gather(*[_one(v) for v in values]))


# ---------------------------------------------------------------------------
# Interface sincrona para scripts de pesquisa
# ---------------------------------------------------------------------------
def run_sync(circuit: dict[str, Any], url: str = DEFAULT_URL,
             priority: int = 5, timeout: float = 300.0) -> dict[str, Any]:
    async def _go() -> dict:
        async with QSimClient(url) as client:
            return await client.run(circuit, priority=priority,
                                    timeout=timeout)
    return asyncio.run(_go())


def sweep_sync(make_circuit: Callable[[Any], dict], values: Iterable[Any],
               url: str = DEFAULT_URL, concurrency: int = 4,
               timeout: float = 300.0) -> list[dict[str, Any]]:
    async def _go() -> list[dict]:
        async with QSimClient(url) as client:
            return await client.sweep(make_circuit, values,
                                      concurrency=concurrency,
                                      timeout=timeout)
    return asyncio.run(_go())
