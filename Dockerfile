FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py README.md LICENSE /app/
COPY coding_agent /app/coding_agent

RUN pip install --upgrade pip && \
    pip install -e ".[math]"

COPY benchmarks /app/benchmarks
COPY scripts /app/scripts
COPY configs /app/configs
COPY examples /app/examples

CMD ["coding-agent", "--help"]
