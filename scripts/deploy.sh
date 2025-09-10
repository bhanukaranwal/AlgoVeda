#!/bin/bash
# AlgoVeda Production Deployment Script
# Automated deployment with health checks and rollback capabilities

set -euo pipefail

# Configuration
PROJECT_NAME="algoveda"
VERSION="${1:-latest}"
ENVIRONMENT="${2:-production}"
BACKUP_COUNT=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Pre-deployment checks
check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || { error "Docker is required but not installed."; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { error "Docker Compose is required but not installed."; exit 1; }
    
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml not found in current directory"
        exit 1
    fi
    
    if [ ! -f ".env.${ENVIRONMENT}" ]; then
        error ".env.${ENVIRONMENT} file not found"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Database backup
backup_database() {
    log "Creating database backup..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # PostgreSQL backup
    docker-compose exec -T postgres pg_dump -U algoveda algoveda | gzip > "$backup_dir/postgres_backup.sql.gz"
    
    # Redis backup
    docker-compose exec -T redis redis-cli --rdb /data/dump.rdb
    docker cp "$(docker-compose ps -q redis):/data/dump.rdb" "$backup_dir/redis_dump.rdb"
    
    log "Database backup created in $backup_dir"
}

# Build and deploy
deploy_services() {
    log "Starting deployment of AlgoVeda $VERSION..."
    
    # Copy environment file
    cp ".env.${ENVIRONMENT}" .env
    
    # Build images
    log "Building Docker images..."
    docker-compose build --parallel --no-cache
    
    # Stop existing services gracefully
    log "Stopping existing services..."
    docker-compose down --remove-orphans
    
    # Start new services
    log "Starting new services..."
    docker-compose up -d
    
    # Wait for services to be ready
    wait_for_services
}

# Health check
wait_for_services() {
    log "Waiting for services to be healthy..."
    
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if check_service_health; then
            log "All services are healthy"
            return 0
        fi
        
        echo -n "."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    error "Services failed to become healthy within timeout"
    return 1
}

check_service_health() {
    # Check AlgoVeda core service
    if ! curl -sf http://localhost:8080/health >/dev/null 2>&1; then
        return 1
    fi
    
    # Check database connectivity
    if ! docker-compose exec -T postgres pg_isready -U algoveda >/dev/null 2>&1; then
        return 1
    fi
    
    # Check Redis
    if ! docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        return 1
    fi
    
    return 0
}

# Database migration
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    docker-compose exec algoveda-core wait-for-it postgres:5432 --timeout=30
    
    # Run migrations
    docker-compose exec algoveda-core python -c "
from algoveda.database import init_db, upgrade_db
init_db()
upgrade_db()
print('Database migrations completed successfully')
"
}

# Post-deployment verification
verify_deployment() {
    log "Verifying deployment..."
    
    # Test API endpoints
    local endpoints=(
        "GET http://localhost:8080/health"
        "GET http://localhost:8080/api/v1/ping"
        "GET http://localhost:8080/metrics"
    )
    
    for endpoint in "${endpoints[@]}"; do
        read -r method url <<< "$endpoint"
        if ! curl -sf -X "$method" "$url" >/dev/null; then
            error "Failed to verify endpoint: $method $url"
            return 1
        fi
    done
    
    # Test WebSocket connection
    if ! curl --include --no-buffer --header "Connection: Upgrade" --header "Upgrade: websocket" --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" --header "Sec-WebSocket-Version: 13" http://localhost:8080/ws >/dev/null 2>&1; then
        warn "WebSocket connection test failed"
    fi
    
    log "Deployment verification passed"
}

# Rollback function
rollback() {
    error "Deployment failed. Initiating rollback..."
    
    # Stop failed deployment
    docker-compose down
    
    # Get latest backup
    local latest_backup=$(ls -1t backups/ | head -1)
    if [ -n "$latest_backup" ]; then
        log "Restoring from backup: $latest_backup"
        
        # Restore database
        gunzip -c "backups/$latest_backup/postgres_backup.sql.gz" | docker-compose exec -T postgres psql -U algoveda algoveda
        docker cp "backups/$latest_backup/redis_dump.rdb" "$(docker-compose ps -q redis):/data/dump.rdb"
        
        # Restart services with previous version
        docker-compose up -d
    fi
    
    error "Rollback completed"
}

# Cleanup old backups
cleanup_backups() {
    log "Cleaning up old backups (keeping $BACKUP_COUNT)..."
    
    if [ -d "backups" ]; then
        ls -1t backups/ | tail -n +$((BACKUP_COUNT + 1)) | xargs -r -I {} rm -rf "backups/{}"
    fi
}

# Performance monitoring
start_monitoring() {
    log "Starting performance monitoring..."
    
    # Create monitoring dashboard
    docker-compose exec grafana grafana-cli admin reset-admin-password admin123
    
    # Import dashboards
    for dashboard in monitoring/dashboards/*.json; do
        if [ -f "$dashboard" ]; then
            curl -X POST \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer admin:admin123" \
                -d "@$dashboard" \
                http://localhost:3000/api/dashboards/db
        fi
    done
}

# Security scan
security_scan() {
    log "Running security scan..."
    
    # Container security scan
    if command -v docker-bench-security >/dev/null 2>&1; then
        docker-bench-security
    fi
    
    # Vulnerability scan
    if command -v trivy >/dev/null 2>&1; then
        trivy image "${PROJECT_NAME}:${VERSION}"
    fi
}

# Main deployment flow
main() {
    log "Starting AlgoVeda deployment..."
    log "Version: $VERSION"
    log "Environment: $ENVIRONMENT"
    
    # Trap for rollback on failure
    trap 'rollback' ERR
    
    check_prerequisites
    backup_database
    deploy_services
    run_migrations
    verify_deployment
    start_monitoring
    cleanup_backups
    
    # Remove trap after successful deployment
    trap - ERR
    
    log "AlgoVeda deployment completed successfully!"
    log "Access the platform at: http://localhost:8080"
    log "Grafana dashboard: http://localhost:3000 (admin/admin123)"
    log "Jupyter notebook: http://localhost:8888"
    
    # Optional security scan
    if [ "${SECURITY_SCAN:-false}" = "true" ]; then
        security_scan
    fi
}

# Script execution
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
