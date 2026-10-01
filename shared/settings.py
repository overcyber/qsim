"""Configuracoes centralizadas do QSim (pydantic-settings, Pydantic >= 2.4)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QSIM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mensageria (NATS). A URL aceita credenciais: nats://user:senha@host:4222
    nats_url: str = "nats://127.0.0.1:4222"
    nats_stream: str = "QSIM_JOBS"
    use_jetstream: bool = True              # fila de trabalho duravel (fallback: core NATS)
    subject_submit: str = "qsim.jobs.submit"
    subject_jobs: str = "qsim.jobs.exec"    # prefixo; scheduler publica em .<classe>
    subject_results: str = "qsim.jobs.result"
    subject_telemetry: str = "qsim.telemetry"
    subject_control: str = "qsim.control"
    engine_queue_group: str = "engines"

    # Scheduler
    max_active_per_origin: int = 8          # quota de jobs simultaneos por origem
    class_medium_min: int = 21              # small: <= 20 qubits
    class_large_min: int = 30               # medium: 21-29; large: 30-31

    # Limites do simulador (defesa em profundidade: tambem no worker)
    max_qubits: int = 31
    max_shots: int = 1_000_000
    max_operations: int = 20_000
    job_timeout_s: int = 300

    # Engine
    engine_backend: str = "auto"            # numpy | cupy | auto
    engine_default_precision: str = "complex128"
    engine_classes: str = "auto"            # auto ou lista: small,medium,large,noisy
    trajectory_workers: int = 0             # 0 = os.cpu_count()

    # Persistencia do gateway (historico de jobs)
    db_path: str = "data/qsim.db"

    # Gateway de administracao (FastAPI)
    admin_host: str = "0.0.0.0"
    admin_port: int = 10000
    admin_token: str = ""                   # vazio = sem autenticacao (apenas dev)
    rate_limit_per_minute: int = 240        # por IP de origem
    max_body_bytes: int = 1_048_576         # 1 MiB por requisicao

    # Metricas Prometheus (porta HTTP de scrape por servico)
    metrics_port_engine: int = 9101
    metrics_port_gateway: int = 9102
    metrics_port_scheduler: int = 9103

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    service_name: str = "qsim"


settings = Settings()
