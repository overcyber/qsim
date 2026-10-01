"""Camada de mensageria do QSim.

Broker: NATS (binario unico ~18 MB, latencia sub-milissegundo, pub/sub,
request-reply e filas de trabalho nativas via queue groups). JetStream
opcional para persistencia de jobs.

Serializacao: MessagePack (binario, 3-10x mais rapido que JSON para os
payloads tipicos de circuito/resultado). Nada de HTTP entre servicos:
toda a comunicacao interna passa pelo NATS sobre TCP puro.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import msgpack
import nats
from nats.aio.client import Client as NatsClient
from nats.aio.msg import Msg


def pack(data: Any) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


def unpack(payload: bytes) -> Any:
    return msgpack.unpackb(payload, raw=False)


class Bus:
    """Wrapper fino sobre o cliente NATS com MessagePack embutido."""

    def __init__(self, url: str, name: str = "qsim"):
        self.url = url
        self.name = name
        self.nc: NatsClient | None = None

    async def connect(self, max_attempts: int = 30, delay_s: float = 1.0) -> None:
        last_err: Exception | None = None
        for _ in range(max_attempts):
            try:
                self.nc = await nats.connect(
                    self.url,
                    name=self.name,
                    max_reconnect_attempts=-1,
                    reconnect_time_wait=2,
                )
                return
            except Exception as err:
                last_err = err
                await asyncio.sleep(delay_s)
        raise ConnectionError(
            f"Nao foi possivel conectar ao NATS em {self.url}: {last_err}"
        )

    async def close(self) -> None:
        if self.nc:
            await self.nc.drain()

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------
    async def publish(self, subject: str, data: Any) -> None:
        assert self.nc is not None, "Bus nao conectado"
        await self.nc.publish(subject, pack(data))

    async def subscribe(
        self,
        subject: str,
        handler: Callable[[str, Any], Awaitable[None]],
        queue: str = "",
    ):
        """Assina um subject. Com queue group, mensagens sao distribuidas
        entre os workers do grupo (balanceamento de carga nativo)."""
        assert self.nc is not None, "Bus nao conectado"

        async def _cb(msg: Msg) -> None:
            await handler(msg.subject, unpack(msg.data))

        return await self.nc.subscribe(subject, queue=queue, cb=_cb)

    # ------------------------------------------------------------------
    # Request-Reply (substitui chamadas HTTP internas)
    # ------------------------------------------------------------------
    async def request(self, subject: str, data: Any, timeout: float = 5.0) -> Any:
        assert self.nc is not None, "Bus nao conectado"
        msg = await self.nc.request(subject, pack(data), timeout=timeout)
        return unpack(msg.data)

    async def respond_to(
        self,
        subject: str,
        handler: Callable[[Any], Awaitable[Any]],
        queue: str = "",
    ):
        """Registra um responder para request-reply."""
        assert self.nc is not None, "Bus nao conectado"

        async def _cb(msg: Msg) -> None:
            result = await handler(unpack(msg.data))
            if msg.reply:
                await msg.respond(pack(result))

        return await self.nc.subscribe(subject, queue=queue, cb=_cb)

    # ------------------------------------------------------------------
    # JetStream: fila de trabalho duravel (sobrevive a reinicios; jobs em
    # execucao quando um worker morre sao reentregues a outro worker).
    # Fallback transparente para core NATS quando JetStream esta desativado
    # no broker.
    # ------------------------------------------------------------------
    async def ensure_workqueue(self, stream: str, subjects: list[str]) -> bool:
        assert self.nc is not None, "Bus nao conectado"
        try:
            from nats.js.api import RetentionPolicy, StreamConfig
            js = self.nc.jetstream()
            try:
                await js.add_stream(StreamConfig(
                    name=stream,
                    subjects=subjects,
                    retention=RetentionPolicy.WORK_QUEUE,
                ))
            except Exception:
                await js.stream_info(stream)  # ja existe
            return True
        except Exception:
            return False

    async def subscribe_workqueue(
        self,
        subject: str,
        handler: Callable[[str, Any], Awaitable[None]],
        queue: str,
        durable: str,
        stream: str,
    ):
        """Consome de uma work queue JetStream (ack apos processar).
        Se JetStream nao estiver disponivel, cai para queue group core."""
        assert self.nc is not None, "Bus nao conectado"
        try:
            js = self.nc.jetstream()

            async def _cb(msg) -> None:
                try:
                    await handler(msg.subject, unpack(msg.data))
                finally:
                    await msg.ack()

            return await js.subscribe(
                subject, queue=queue, durable=durable, stream=stream,
                cb=_cb, manual_ack=True,
            )
        except Exception:
            return await self.subscribe(subject, handler, queue=queue)
