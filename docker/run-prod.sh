#!/bin/bash

# Production startup script for Video Translator
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    fi
    print_status "Docker is running"
}

# Check if required ports are available
check_ports() {
    local ports=("80")
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_warning "Port $port is already in use. Please stop the service using port $port and try again."
            exit 1
        fi
    done
    print_status "Required ports are available"
}

# Create necessary directories
create_directories() {
    local dirs=("uploads" "results" "logs")
    for dir in "${dirs[@]}"; do
        if [ ! -d "../$dir" ]; then
            mkdir -p "../$dir"
            print_status "Created directory: $dir"
        fi
    done
}

# Build and start services
start_services() {
    print_status "Building and starting production services..."
    
    # Build images
    print_status "Building backend image..."
    docker-compose -f docker-compose.prod.yml build backend
    
    print_status "Building frontend image..."
    docker-compose -f docker-compose.prod.yml build frontend
    
    # Start services
    print_status "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    timeout=120
    counter=0
    
    while [ $counter -lt $timeout ]; do
        if docker-compose -f docker-compose.prod.yml ps | grep -q "healthy"; then
            print_status "All services are healthy!"
            break
        fi
        sleep 2
        counter=$((counter + 2))
        echo -n "."
    done
    
    if [ $counter -eq $timeout ]; then
        print_error "Services failed to become healthy within $timeout seconds"
        docker-compose -f docker-compose.prod.yml logs
        exit 1
    fi
}

# Show service status
show_status() {
    print_status "Service Status:"
    docker-compose -f docker-compose.prod.yml ps
    
    print_status "Application URLs:"
    echo "  Frontend: http://localhost"
    echo "  Backend API: http://localhost/api"
    echo "  Health Check: http://localhost/health"
    
    print_status "Logs can be viewed with:"
    echo "  docker-compose -f docker-compose.prod.yml logs -f"
}

# Main execution
main() {
    print_status "Starting Video Translator in production mode..."
    
    # Change to docker directory
    cd "$(dirname "$0")"
    
    check_docker
    check_ports
    create_directories
    start_services
    show_status
    
    print_status "Video Translator is now running in production mode!"
    print_warning "For production deployment, consider:"
    echo "  - Setting up SSL certificates"
    echo "  - Configuring a domain name"
    echo "  - Setting up monitoring and logging"
    echo "  - Configuring backup strategies"
}

# Handle script interruption
trap 'print_error "Script interrupted. Stopping services..."; docker-compose -f docker-compose.prod.yml down; exit 1' INT TERM

# Run main function
main "$@" 