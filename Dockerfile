FROM --platform=linux/amd64 python:3.11-slim

LABEL maintainer="liaici"
LABEL description="DuckDB + spatial pour Liaici"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

ARG DUCKDB_VERSION=1.5.5
RUN curl -L "https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/duckdb_cli-linux-amd64.zip" \
    -o /tmp/duckdb.zip \
    && unzip /tmp/duckdb.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/duckdb \
    && rm /tmp/duckdb.zip

RUN pip install --no-cache-dir duckdb==1.5.5

WORKDIR /app
RUN mkdir -p /app/data/parquet /app/data/db /app/scripts

RUN duckdb -c "INSTALL spatial; INSTALL httpfs; INSTALL json;"

COPY scripts/00_init.sql /app/scripts/00_init.sql
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]