# Multi-Stage Production Docker Build for AlgoVeda Trading Platform
# Optimized for security, performance, and minimal image size

# Stage 1: Rust Build Environment
FROM rust:1.70-slim as rust-builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Copy Cargo files for dependency caching
COPY core-engine/Cargo.toml core-engine/Cargo.lock ./
COPY core-engine/src ./src

# Build dependencies separately for better caching
RUN cargo build --release --bins
RUN cargo build --release

# Stage 2: Go Build Environment
FROM golang:1.21-alpine as go-builder

# Install git for go modules
RUN apk add --no-cache git

WORKDIR /app

# Copy Go modules files
COPY websocket-gateway/go.mod websocket-gateway/go.sum ./
RUN go mod download

# Copy source code
COPY websocket-gateway/ .

# Build the Go application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o websocket-gateway ./cmd/main.go

# Stage 3: Node.js Build Environment
FROM node:18-alpine as node-builder

WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --only=production

# Copy source and build
COPY frontend/ .
RUN npm run build

# Stage 4: Python Environment for Backtesting
FROM python:3.11-slim as python-builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY backtesting-engine/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy Python source
COPY backtesting-engine/ .

# Stage 5: Runtime Environment
FROM ubuntu:22.04

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd -m -u 1001 -s /bin/bash appuser

# Create application directories
RUN mkdir -p /app/bin /app/config /app/logs /app/data \
    && chown -R appuser:appuser /app

# Copy built binaries from build stages
COPY --from=rust-builder --chown=appuser:appuser /app/target/release/algoveda /app/bin/
COPY --from=go-builder --chown=appuser:appuser /app/websocket-gateway /app/bin/
COPY --from=node-builder --chown=appuser:appuser /app/build /app/frontend/
COPY --from=python-builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --from=python-builder --chown=appuser:appuser /app /app/backtesting/

# Copy configuration files
COPY --chown=appuser:appuser configs/ /app/config/

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV PYTHONPATH=/app/backtesting

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8080 8081 9090

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start services in background\n\
echo "Starting AlgoVeda Core Engine..."\n\
/app/bin/algoveda &\n\
CORE_PID=$!\n\
\n\
echo "Starting WebSocket Gateway..."\n\
/app/bin/websocket-gateway &\n\
WS_PID=$!\n\
\n\
# Function to handle shutdown\n\
shutdown() {\n\
    echo "Shutting down services..."\n\
    kill $CORE_PID $WS_PID 2>/dev/null || true\n\
    wait $CORE_PID $WS_PID 2>/dev/null || true\n\
    echo "Services stopped."\n\
    exit 0\n\
}\n\
\n\
# Trap signals\n\
trap shutdown SIGTERM SIGINT\n\
\n\
# Wait for services\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

# Labels for metadata
LABEL maintainer="AlgoVeda Team <team@algoveda.com>"
LABEL version="1.0.0"
LABEL description="AlgoVeda Algorithmic Trading Platform"
LABEL org.opencontainers.image.source="https://github.com/algoveda/platform"
