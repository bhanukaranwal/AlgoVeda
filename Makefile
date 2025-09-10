# AlgoVeda Trading Platform Makefile
# Complete build system for Rust, C++, CUDA, and related components

.PHONY: all clean build test release install docker deploy docs benchmark

# Variables
CARGO_FLAGS = --release
CMAKE_BUILD_TYPE = Release
DOCKER_TAG = algoveda/platform:latest
TEST_THREADS = 4

# Default target
all: build

# Build all components
build: build-rust build-cpp build-cuda build-frontend

# Build Rust components
build-rust:
	@echo "Building Rust core engine..."
	cd core-engine && cargo build $(CARGO_FLAGS)
	@echo "Building WebSocket gateway..."
	cd websocket-gateway && cargo build $(CARGO_FLAGS)

# Build C++ components
build-cpp:
	@echo "Building C++ performance components..."
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) ../cpp
	cd build && make -j$(shell nproc)

# Build CUDA components
build-cuda:
	@echo "Building CUDA GPU kernels..."
	cd cuda && mkdir -p build
	cd cuda/build && cmake -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) ..
	cd cuda/build && make -j$(shell nproc)

# Build frontend
build-frontend:
	@echo "Building React/TypeScript frontend..."
	cd frontend && npm install
	cd frontend && npm run build

# Run tests
test: test-rust test-cpp test-integration

test-rust:
	@echo "Running Rust tests..."
	cd core-engine && cargo test $(CARGO_FLAGS) -- --test-threads=$(TEST_THREADS)
	cd websocket-gateway && cargo test $(CARGO_FLAGS)

test-cpp:
	@echo "Running C++ tests..."
	cd build && ctest --verbose

test-integration:
	@echo "Running integration tests..."
	cd tests/integration && cargo test $(CARGO_FLAGS)

# Performance benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	cd core-engine && cargo bench
	cd benchmarks && cargo run --release

# Release build with optimizations
release: CARGO_FLAGS = --release
release: CMAKE_BUILD_TYPE = Release
release: build
	@echo "Creating optimized release build..."
	strip core-engine/target/release/algoveda
	strip websocket-gateway/target/release/websocket-gateway

# Install system dependencies
install-deps:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake pkg-config
	sudo apt-get install -y libssl-dev libpq-dev redis-server
	sudo apt-get install -y nvidia-cuda-toolkit # For CUDA support
	cargo install cargo-audit cargo-outdated

# Database setup
db-setup:
	@echo "Setting up PostgreSQL database..."
	sudo -u postgres createdb algoveda || true
	sudo -u postgres psql -d algoveda -f sql/init.sql

# Docker build
docker:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_TAG) .

# Docker compose for development
docker-dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up --build

# Deploy to production
deploy: release docker
	@echo "Deploying to production..."
	./scripts/deploy.sh

# Generate documentation
docs:
	@echo "Generating documentation..."
	cd core-engine && cargo doc --no-deps
	cd frontend && npm run docs

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cd core-engine && cargo clean
	cd websocket-gateway && cargo clean
	rm -rf build/
	rm -rf cuda/build/
	cd frontend && rm -rf build/ node_modules/

# Security audit
audit:
	@echo "Running security audit..."
	cd core-engine && cargo audit
	cd websocket-gateway && cargo audit
	cd frontend && npm audit

# Format code
fmt:
	@echo "Formatting code..."
	cd core-engine && cargo fmt
	cd websocket-gateway && cargo fmt
	cd frontend && npm run format

# Lint code
lint:
	@echo "Linting code..."
	cd core-engine && cargo clippy -- -D warnings
	cd websocket-gateway && cargo clippy -- -D warnings
	cd frontend && npm run lint

# Check dependencies
deps-check:
	@echo "Checking for outdated dependencies..."
	cd core-engine && cargo outdated
	cd frontend && npm outdated

# Performance profiling
profile:
	@echo "Running performance profiling..."
	cd core-engine && cargo build --profile=profiling
	perf record -g ./target/profiling/algoveda
	perf report

# Load testing
load-test:
	@echo "Running load tests..."
	cd tests/load && ./run_load_tests.sh

# Continuous integration
ci: install-deps build test audit lint
	@echo "CI pipeline completed successfully"

# Show help
help:
	@echo "AlgoVeda Trading Platform Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all components (default)"
	@echo "  build        - Build all components"
	@echo "  build-rust   - Build only Rust components"
	@echo "  build-cpp    - Build only C++ components"
	@echo "  build-cuda   - Build only CUDA components"
	@echo "  build-frontend - Build only frontend"
	@echo "  test         - Run all tests"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  release      - Create optimized release build"
	@echo "  docker       - Build Docker image"
	@echo "  deploy       - Deploy to production"
	@echo "  docs         - Generate documentation"
	@echo "  clean        - Clean build artifacts"
	@echo "  audit        - Run security audit"
	@echo "  fmt          - Format code"
	@echo "  lint         - Lint code"
	@echo "  help         - Show this help message"

# Version information
version:
	@echo "AlgoVeda Trading Platform v1.0.0"
	@echo "Rust version: $(shell rustc --version)"
	@echo "GCC version: $(shell gcc --version | head -n1)"
	@echo "Node.js version: $(shell node --version)"
	@echo "Docker version: $(shell docker --version)"
