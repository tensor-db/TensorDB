# Multi-stage Dockerfile for TensorDB server
# Build: docker build -t tensordb .
# Run:   docker run -p 5433:5433 -v tensordb_data:/data tensordb

# ── Build stage ────────────────────────────────────────────────────────────

FROM rust:1.82-bookworm AS builder

WORKDIR /app
COPY . .

# Build server and CLI without LLM feature (smaller image)
RUN cargo build --release -p tensordb-server -p tensordb-cli --no-default-features \
    && strip target/release/tensordb-server target/release/tensordb-cli

# ── Runtime stage ──────────────────────────────────────────────────────────

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/tensordb-server /usr/local/bin/
COPY --from=builder /app/target/release/tensordb-cli /usr/local/bin/

# Data directory
RUN mkdir -p /data
VOLUME /data

# PostgreSQL wire protocol port
EXPOSE 5433

ENV TENSORDB_DATA_DIR=/data
ENV TENSORDB_PORT=5433

ENTRYPOINT ["tensordb-server"]
CMD ["--data-dir", "/data", "--port", "5433"]
