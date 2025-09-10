#!/bin/bash
# Zero-Downtime Production Deployment Script for AlgoVeda

set -euo pipefail

# Configuration
NAMESPACE="algoveda-prod"
IMAGE_TAG="${1:-latest}"
DEPLOYMENT_NAME="algoveda-core-engine"
TIMEOUT="600s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Pre-deployment checks
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl connectivity
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace $NAMESPACE >/dev/null 2>&1; then
        error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check image exists
    if ! docker manifest inspect algoveda/platform:$IMAGE_TAG >/dev/null 2>&1; then
        error "Image algoveda/platform:$IMAGE_TAG does not exist"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Health check function
health_check() {
    local endpoint=$1
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$endpoint/health" >/dev/null 2>&1; then
            return 0
        fi
        log "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 2
        ((attempt++))
    done
    return 1
}

# Deploy with rolling update
deploy() {
    log "Starting deployment of image: algoveda/platform:$IMAGE_TAG"
    
    # Update deployment image
    kubectl set image deployment/$DEPLOYMENT_NAME \
        core-engine=algoveda/platform:$IMAGE_TAG \
        websocket-gateway=algoveda/websocket-gateway:$IMAGE_TAG \
        -n $NAMESPACE
    
    log "Waiting for rollout to complete..."
    kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=$TIMEOUT
    
    # Verify deployment
    local ready_replicas=$(kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
    local desired_replicas=$(kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.spec.replicas}')
    
    if [ "$ready_replicas" -eq "$desired_replicas" ]; then
        log "Deployment successful: $ready_replicas/$desired_replicas replicas ready"
    else
        error "Deployment failed: only $ready_replicas/$desired_replicas replicas ready"
        return 1
    fi
}

# Run smoke tests
smoke_tests() {
    log "Running smoke tests..."
    
    # Get service endpoint
    local service_ip=$(kubectl get service algoveda-core-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress.ip}')
    if [ -z "$service_ip" ]; then
        service_ip=$(kubectl get service algoveda-core-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    fi
    
    local endpoint="http://$service_ip:8080"
    
    # Health check
    if health_check "$endpoint"; then
        log "Health check passed"
    else
        error "Health check failed"
        return 1
    fi
    
    # API connectivity test
    if curl -f -s "$endpoint/api/health" >/dev/null 2>&1; then
        log "API connectivity test passed"
    else
        warn "API connectivity test failed"
    fi
    
    # WebSocket connectivity test
    if curl -f -s "$endpoint:8081/ws" >/dev/null 2>&1; then
        log "WebSocket connectivity test passed"
    else
        warn "WebSocket connectivity test failed"
    fi
    
    log "Smoke tests completed"
}

# Rollback function
rollback() {
    error "Deployment failed, initiating rollback..."
    
    kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE
    kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=$TIMEOUT
    
    log "Rollback completed"
}

# Cleanup old replica sets
cleanup() {
    log "Cleaning up old replica sets..."
    
    kubectl delete replicaset -n $NAMESPACE \
        -l app=$DEPLOYMENT_NAME \
        --field-selector=status.replicas=0
    
    log "Cleanup completed"
}

# Main deployment flow
main() {
    log "Starting AlgoVeda production deployment"
    
    check_prerequisites
    
    if deploy; then
        if smoke_tests; then
            cleanup
            log "Deployment completed successfully!"
        else
            warn "Smoke tests failed, but deployment is running"
        fi
    else
        rollback
        exit 1
    fi
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"
