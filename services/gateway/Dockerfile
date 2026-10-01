FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY services/__init__.py ./services/__init__.py
COPY services/gateway ./services/gateway

ENV PYTHONUNBUFFERED=1
EXPOSE 10000
CMD ["python", "-m", "services.gateway.main"]
