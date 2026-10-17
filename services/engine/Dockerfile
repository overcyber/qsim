FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# libgomp1 do sistema: a copia embutida no wheel do qiskit-aer falha com
# "cannot allocate memory in static TLS block" na imagem slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt \
    && echo /usr/local/lib/python3.12/site-packages/qiskit_aer.libs/libgomp-*.so.* \
        > /etc/ld.so.preload
# Preload da libgomp: a copia manylinux do wheel usa TLS initial-exec e, se
# for dlopen'ada tardiamente pelo Aer, falha com "cannot allocate memory in
# static TLS block" na imagem slim.

COPY shared ./shared
COPY services/__init__.py ./services/__init__.py
COPY services/engine ./services/engine

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "services.engine.worker"]
