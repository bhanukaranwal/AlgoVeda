# Multi-stage Dockerfile for AlgoVeda Trading Platform
# Builds all components in optimized production container

# Base image with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=${CUDA_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    wget \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    libssl-dev \
    libpq-dev \
    redis-tools \
    valgrind \
    perf-tools-unstable \
    libclang-dev \
    llvm-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain stable
RUN rustup component add clippy rustfmt
RUN cargo install cargo-audit cargo-tarpaulin

# Install Go
ENV GO_VERSION=1.21.0
RUN wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz \
    && rm go${GO_VERSION}.linux-amd64.tar.gz
ENV PATH=/usr/local/go/bin:$PATH
ENV GOPATH=/go
ENV PATH=$GOPATH/bin:$PATH

# Install Node.js and npm for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel numpy pandas scipy matplotlib seaborn \
    jupyter notebook jupyterlab plotly dash streamlit fastapi uvicorn pydantic sqlalchemy \
    pytest pytest-cov flake8 mypy sphinx black isort

# Set working directory
WORKDIR /algoveda

# Copy source code
COPY . .

# Build stage: Compile all components
FROM builder as build

# Build Rust core
WORKDIR /algoveda/core-engine
RUN cargo build --release --target x86_64-unknown-linux-gnu

# Build C++ components
WORKDIR /algoveda
RUN mkdir -p build && \
    cd cpp && \
    cmake -B ../build/cpp -S . -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math" \
        -DWITH_CUDA=ON \
        -DWITH_SIMD=ON \
        -DWITH_OPENMP=ON && \
    ninja -C ../build/cpp

# Build CUDA kernels
WORKDIR /algoveda/cuda
RUN mkdir -p ../build/cuda && \
    nvcc -O3 --gpu-architecture=sm_75 --use_fast_math --shared --compiler-options '-fPIC' \
        -I include -I /usr/local/cuda/include \
        src/kernels/*.cu -o ../build/cuda/libalgoveda_cuda.so \
        -lcuda -lcudart -lcublas -lcurand

# Build Go WebSocket gateway
WORKDIR /algoveda/websocket-gateway
RUN CGO_ENABLED=1 go build -a -installsuffix cgo -ldflags="-s -w" -o ../build/websocket-gateway ./cmd

# Build Python extensions
WORKDIR /algoveda/src/python
RUN python3 setup.py build_ext --inplace && \
    python3 setup.py bdist_wheel

# Build TypeScript frontend
WORKDIR /algoveda/frontend
RUN npm install && \
    npm run build:prod

# Production stage: Create minimal runtime image
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    redis-tools \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -m -u 1000 algoveda && \
    mkdir -p /app /data /logs && \
    chown -R algoveda:algoveda /app /data /logs

# Set working directory
WORKDIR /app

# Copy built artifacts from build stage
COPY --from=build --chown=algoveda:algoveda /algoveda/target/release/libalgoveda_core.so /usr/local/lib/
COPY --from=build --chown=algoveda:algoveda /algoveda/build/cpp/libalgoveda_cpp.so /usr/local/lib/
COPY --from=build --chown=algoveda:algoveda /algoveda/build/cuda/libalgoveda_cuda.so /usr/local/lib/
COPY --from=build --chown=algoveda:algoveda /algoveda/build/websocket-gateway /app/
COPY --from=build --chown=algoveda:algoveda /algoveda/src/python/dist/*.whl /tmp/
COPY --from=build --chown=algoveda:algoveda /algoveda/frontend/dist /app/frontend/

# Copy configuration files
COPY --chown=algoveda:algoveda configs/ /app/configs/
COPY --chown=algoveda:algoveda scripts/ /app/scripts/

# Install Python wheel
RUN python3 -m pip install /tmp/*.whl && rm -rf /tmp/*.whl

# Update library cache
RUN ldconfig

# Switch to application user
USER algoveda

# Environment variables
ENV ALGOVEDA_CONFIG_PATH=/app/configs/production.yaml
ENV ALGOVEDA_DATA_PATH=/data
ENV ALGOVEDA_LOG_PATH=/logs
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8443 9090

# Volume for persistent data
VOLUME ["/data", "/logs"]

# Default command
CMD ["/app/websocket-gateway", "-config", "/app/configs/production.yaml"]

# Development stage: Include development tools
FROM build as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    strace \
    htop \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Go development tools
RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest && \
    go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest

# Install Jupyter and additional Python packages
RUN python3 -m pip install jupyter jupyterlab ipywidgets

# Set development environment
ENV ENVIRONMENT=development
ENV LOG_LEVEL=debug

# Expose additional ports for development
EXPOSE 8888 9000

# Development command
CMD ["bash"]

# Testing stage: Run all tests
FROM development as testing

WORKDIR /algoveda

# Copy test data and scripts
COPY tests/ tests/
COPY scripts/run-tests.sh scripts/

# Run tests
RUN chmod +x scripts/run-tests.sh && \
    scripts/run-tests.sh

# Benchmark stage: Performance testing
FROM testing as benchmark

# Copy benchmark data
COPY benchmarks/ benchmarks/

# Run benchmarks
RUN chmod +x scripts/run-benchmarks.sh && \
    scripts/run-benchmarks.sh > /tmp/benchmark-results.txt

# Documentation stage: Generate documentation
FROM development as docs

# Install documentation tools
RUN python3 -m pip install sphinx sphinx-rtd-theme myst-parser
RUN cargo install mdbook

# Generate documentation
RUN cd core-engine && cargo doc --no-deps --document-private-items
RUN cd src/python && sphinx-build -b html docs /tmp/docs/python
RUN cd docs && mdbook build

# Copy generated docs
COPY --from=docs /tmp/docs /app/docs/

# Multi-architecture support
FROM --platform=$BUILDPLATFORM production as final

# Build arguments for multi-arch
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Label the image
LABEL org.opencontainers.image.title="AlgoVeda Trading Platform"
LABEL org.opencontainers.image.description="Complete algorithmic trading platform"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="AlgoVeda"
LABEL org.opencontainers.image.source="https://github.com/algoveda/platform"
LABEL org.opencontainers.image.platform=$TARGETPLATFORM

# Final stage
FROM final
