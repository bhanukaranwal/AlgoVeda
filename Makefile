# AlgoVeda Complete Trading Platform Makefile
# Builds all components: Rust core, C++ modules, CUDA kernels, Go gateway, Python extensions

# Project configuration
PROJECT_NAME := algoveda
VERSION := 1.0.0
BUILD_DIR := build
DIST_DIR := dist
TARGET_DIR := target

# Compiler configurations
RUST_TARGET := x86_64-unknown-linux-gnu
CC := gcc
CXX := g++
NVCC := nvcc
GO := go

# Optimization flags
RUST_FLAGS := --release
CXX_FLAGS := -std=c++20 -O3 -march=native -mtune=native -ffast-math -funroll-loops -DNDEBUG
CUDA_FLAGS := -O3 --gpu-architecture=sm_75 --use_fast_math --maxrregcount=64
GO_FLAGS := -ldflags="-s -w" -a -installsuffix cgo

# CUDA configuration
CUDA_PATH := $(shell which nvcc | sed 's/\/bin\/nvcc//')
CUDA_LIBS := -lcuda -lcudart -lcublas -lcurand -lcufft

# Python configuration
PYTHON := python3
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIBS := $(shell $(PYTHON)-config --ldflags)

# Include paths
INCLUDES := -I./cpp/include -I./cuda/include -I$(CUDA_PATH)/include -I$(PYTHON_INCLUDE)

# Library paths
LIBPATHS := -L./cpp/lib -L$(CUDA_PATH)/lib64 -L./target/release

# Source files
RUST_SOURCES := $(shell find src -name "*.rs")
CXX_SOURCES := $(shell find cpp/src -name "*.cpp")
CUDA_SOURCES := $(shell find cuda/src -name "*.cu")
GO_SOURCES := $(shell find websocket-gateway -name "*.go")
PYTHON_SOURCES := $(shell find src/python -name "*.py")

# Object files
CXX_OBJECTS := $(CXX_SOURCES:cpp/src/%.cpp=$(BUILD_DIR)/cpp/%.o)
CUDA_OBJECTS := $(CUDA_SOURCES:cuda/src/%.cu=$(BUILD_DIR)/cuda/%.o)

# Targets
RUST_LIB := $(TARGET_DIR)/release/libalgoveda_core.so
CXX_LIB := $(BUILD_DIR)/libalgoveda_cpp.so
CUDA_LIB := $(BUILD_DIR)/libalgoveda_cuda.so
GO_BINARY := $(BUILD_DIR)/websocket-gateway
PYTHON_EXT := $(BUILD_DIR)/algoveda_python.so

# Default target
.PHONY: all
all: $(RUST_LIB) $(CXX_LIB) $(CUDA_LIB) $(GO_BINARY) $(PYTHON_EXT)

# Create build directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)/cpp $(BUILD_DIR)/cuda $(BUILD_DIR)/go $(BUILD_DIR)/python

# Rust core library
$(RUST_LIB): $(RUST_SOURCES) | $(BUILD_DIR)
	@echo "Building Rust core library..."
	cd core-engine && cargo build $(RUST_FLAGS) --target $(RUST_TARGET)
	@echo "Rust core library built successfully"

# C++ library
$(CXX_LIB): $(CXX_OBJECTS) | $(BUILD_DIR)
	@echo "Linking C++ library..."
	$(CXX) -shared -fPIC $(CXX_OBJECTS) -o $@ $(LIBPATHS) $(CUDA_LIBS)
	@echo "C++ library built successfully"

$(BUILD_DIR)/cpp/%.o: cpp/src/%.cpp | $(BUILD_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -fPIC $(INCLUDES) -c $< -o $@

# CUDA library
$(CUDA_LIB): $(CUDA_OBJECTS) | $(BUILD_DIR)
	@echo "Linking CUDA library..."
	$(NVCC) -shared $(CUDA_OBJECTS) -o $@ $(CUDA_LIBS)
	@echo "CUDA library built successfully"

$(BUILD_DIR)/cuda/%.o: cuda/src/%.cu | $(BUILD_DIR)
	@echo "Compiling CUDA kernel $<..."
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Go WebSocket gateway
$(GO_BINARY): $(GO_SOURCES) | $(BUILD_DIR)
	@echo "Building Go WebSocket gateway..."
	cd websocket-gateway && CGO_ENABLED=1 $(GO) build $(GO_FLAGS) -o ../$(GO_BINARY) ./cmd
	@echo "Go WebSocket gateway built successfully"

# Python extensions
$(PYTHON_EXT): $(PYTHON_SOURCES) | $(BUILD_DIR)
	@echo "Building Python extensions..."
	cd src/python && $(PYTHON) setup.py build_ext --inplace
	cp src/python/build/lib.*/*.so $(BUILD_DIR)/
	@echo "Python extensions built successfully"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(TARGET_DIR) $(DIST_DIR)
	cd core-engine && cargo clean
	cd websocket-gateway && $(GO) clean
	cd src/python && $(PYTHON) setup.py clean --all
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete
	@echo "Clean completed"

# Run tests
.PHONY: test
test: test-rust test-cpp test-cuda test-go test-python

.PHONY: test-rust
test-rust:
	@echo "Running Rust tests..."
	cd core-engine && cargo test $(RUST_FLAGS)

.PHONY: test-cpp
test-cpp: $(CXX_LIB)
	@echo "Running C++ tests..."
	cd cpp && make test

.PHONY: test-cuda
test-cuda: $(CUDA_LIB)
	@echo "Running CUDA tests..."
	cd cuda && make test

.PHONY: test-go
test-go:
	@echo "Running Go tests..."
	cd websocket-gateway && $(GO) test -v ./...

.PHONY: test-python
test-python: $(PYTHON_EXT)
	@echo "Running Python tests..."
	cd src/python && $(PYTHON) -m pytest tests/ -v

# Benchmarks
.PHONY: benchmark
benchmark: benchmark-rust benchmark-cuda benchmark-python

.PHONY: benchmark-rust
benchmark-rust:
	@echo "Running Rust benchmarks..."
	cd core-engine && cargo bench

.PHONY: benchmark-cuda
benchmark-cuda: $(CUDA_LIB)
	@echo "Running CUDA benchmarks..."
	cd cuda && make benchmark

.PHONY: benchmark-python
benchmark-python: $(PYTHON_EXT)
	@echo "Running Python benchmarks..."
	cd src/python && $(PYTHON) -m pytest benchmarks/ -v

# Documentation
.PHONY: docs
docs: docs-rust docs-python

.PHONY: docs-rust
docs-rust:
	@echo "Generating Rust documentation..."
	cd core-engine && cargo doc --no-deps

.PHONY: docs-python
docs-python:
	@echo "Generating Python documentation..."
	cd src/python && sphinx-build -b html docs $(BUILD_DIR)/docs/python

# Installation
.PHONY: install
install: all
	@echo "Installing AlgoVeda platform..."
	# Install Rust library
	cp $(RUST_LIB) /usr/local/lib/
	# Install C++ library  
	cp $(CXX_LIB) /usr/local/lib/
	# Install CUDA library
	cp $(CUDA_LIB) /usr/local/lib/
	# Install Go binary
	cp $(GO_BINARY) /usr/local/bin/
	# Install Python extensions
	cd src/python && $(PYTHON) setup.py install
	# Update library cache
	ldconfig
	@echo "Installation completed"

# Docker build
.PHONY: docker
docker:
	@echo "Building Docker image..."
	docker build -t $(PROJECT_NAME):$(VERSION) .
	docker tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "Docker image built successfully"

# Create distribution package
.PHONY: dist
dist: all | $(DIST_DIR)
	@echo "Creating distribution package..."
	mkdir -p $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)
	
	# Copy binaries
	cp $(RUST_LIB) $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)/
	cp $(CXX_LIB) $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)/
	cp $(CUDA_LIB) $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)/
	cp $(GO_BINARY) $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)/
	
	# Copy configuration files
	cp -r configs $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)/
	
	# Copy Python package
	cd src/python && $(PYTHON) setup.py sdist --dist-dir ../../$(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)/
	
	# Create archive
	cd $(DIST_DIR) && tar -czf $(PROJECT_NAME)-$(VERSION).tar.gz $(PROJECT_NAME)-$(VERSION)/
	@echo "Distribution package created: $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION).tar.gz"

$(DIST_DIR):
	mkdir -p $(DIST_DIR)

# Performance profiling
.PHONY: profile
profile: $(RUST_LIB)
	@echo "Running performance profiling..."
	cd core-engine && cargo build --release --features=profiling
	perf record --call-graph=dwarf -g ./target/release/examples/benchmark
	perf report

# Memory profiling
.PHONY: memcheck
memcheck: $(RUST_LIB) $(CXX_LIB)
	@echo "Running memory check..."
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
		./target/release/examples/memory_test

# Static analysis
.PHONY: lint
lint: lint-rust lint-go lint-python

.PHONY: lint-rust
lint-rust:
	@echo "Running Rust linting..."
	cd core-engine && cargo clippy -- -D warnings

.PHONY: lint-go
lint-go:
	@echo "Running Go linting..."
	cd websocket-gateway && golangci-lint run

.PHONY: lint-python
lint-python:
	@echo "Running Python linting..."
	cd src/python && flake8 algoveda/ tests/
	cd src/python && mypy algoveda/

# Security audit
.PHONY: audit
audit: audit-rust audit-go audit-python

.PHONY: audit-rust
audit-rust:
	@echo "Running Rust security audit..."
	cd core-engine && cargo audit

.PHONY: audit-go
audit-go:
	@echo "Running Go security audit..."
	cd websocket-gateway && gosec ./...

.PHONY: audit-python
audit-python:
	@echo "Running Python security audit..."
	cd src/python && safety check

# Coverage report
.PHONY: coverage
coverage: coverage-rust coverage-python

.PHONY: coverage-rust
coverage-rust:
	@echo "Generating Rust coverage report..."
	cd core-engine && cargo tarpaulin --out Html --output-dir $(abspath $(BUILD_DIR))/coverage/rust

.PHONY: coverage-python
coverage-python:
	@echo "Generating Python coverage report..."
	cd src/python && coverage run -m pytest tests/
	cd src/python && coverage html -d $(abspath $(BUILD_DIR))/coverage/python

# Continuous integration
.PHONY: ci
ci: clean lint audit test coverage
	@echo "Continuous integration pipeline completed"

# Development setup
.PHONY: dev-setup
dev-setup:
	@echo "Setting up development environment..."
	# Install Rust toolchain
	rustup update stable
	rustup default stable
	rustup component add clippy rustfmt
	cargo install cargo-audit cargo-tarpaulin
	
	# Install Go tools
	$(GO) install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	$(GO) install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest
	
	# Install Python tools
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install flake8 mypy pytest coverage safety sphinx
	
	# Install system dependencies
	sudo apt-get update
	sudo apt-get install -y build-essential pkg-config libssl-dev cuda-toolkit
	
	@echo "Development environment setup completed"

# Help
.PHONY: help
help:
	@echo "AlgoVeda Trading Platform Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all components"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run all tests"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  docs         - Generate documentation"
	@echo "  install      - Install the platform"
	@echo "  docker       - Build Docker image"
	@echo "  dist         - Create distribution package"
	@echo "  profile      - Run performance profiling"
	@echo "  memcheck     - Run memory check"
	@echo "  lint         - Run static analysis"
	@echo "  audit        - Run security audit"
	@echo "  coverage     - Generate coverage reports"
	@echo "  ci           - Run CI pipeline"
	@echo "  dev-setup    - Setup development environment"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Component-specific targets:"
	@echo "  $(RUST_LIB)  - Build Rust core library"
	@echo "  $(CXX_LIB)   - Build C++ library"
	@echo "  $(CUDA_LIB)  - Build CUDA library"
	@echo "  $(GO_BINARY) - Build Go WebSocket gateway"
	@echo "  $(PYTHON_EXT) - Build Python extensions"

# Include component-specific makefiles
-include cpp/Makefile.cpp
-include cuda/Makefile.cuda
-include websocket-gateway/Makefile.go

# Make sure intermediate files are not deleted
.PRECIOUS: $(BUILD_DIR)/cpp/%.o $(BUILD_DIR)/cuda/%.o

# Default shell
SHELL := /bin/bash

# Enable parallel builds
MAKEFLAGS += -j$(shell nproc)
