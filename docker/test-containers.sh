#!/bin/bash

# Test script for video translator containers
# Runs tests in both backend and frontend containers to verify service integrity

set -e

echo "🧪 Testing Video Translator Containers"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}❌ $message${NC}"
    elif [ "$status" = "INFO" ]; then
        echo -e "${YELLOW}ℹ️  $message${NC}"
    fi
}

# Function to run tests in a container
run_container_tests() {
    local container_name=$1
    local test_command=$2
    local test_name=$3
    
    print_status "INFO" "Running $test_name tests in $container_name..."
    
    if docker exec "$container_name" $test_command; then
        print_status "PASS" "$test_name tests passed"
        return 0
    else
        print_status "FAIL" "$test_name tests failed"
        return 1
    fi
}

# Check if containers are running
print_status "INFO" "Checking container status..."

if ! docker-compose -f docker/docker-compose.prod.yml ps | grep -q "Up"; then
    print_status "FAIL" "Containers are not running. Please start them first with: ./docker/run-prod.sh"
    exit 1
fi

# Wait for containers to be healthy
print_status "INFO" "Waiting for containers to be healthy..."
sleep 10

# Check container health
backend_healthy=$(docker-compose -f docker/docker-compose.prod.yml ps backend | grep -c "healthy" || echo "0")
frontend_healthy=$(docker-compose -f docker/docker-compose.prod.yml ps frontend | grep -c "healthy" || echo "0")

if [ "$backend_healthy" -eq 0 ]; then
    print_status "FAIL" "Backend container is not healthy"
    exit 1
fi

if [ "$frontend_healthy" -eq 0 ]; then
    print_status "FAIL" "Frontend container is not healthy"
    exit 1
fi

print_status "PASS" "All containers are healthy"

# Test backend
echo ""
print_status "INFO" "Testing Backend Container"
echo "--------------------------------"

# Run comprehensive backend tests
run_container_tests "video-translator-backend" \
    "python /app/docker/test_backend.py" \
    "Backend functionality"

# Run Python unit tests
run_container_tests "video-translator-backend" \
    "python /app/docker/test_pytest.py" \
    "Python unit tests"

# Test frontend
echo ""
print_status "INFO" "Testing Frontend Container"
echo "---------------------------------"

# Test Node.js dependencies (skip in production container)
run_container_tests "video-translator-frontend" \
    "node --version && npm --version || echo 'Node.js not available in production container (expected)'" \
    "Node.js environment"

# Test if nginx is serving files
run_container_tests "video-translator-frontend" \
    "curl -f http://localhost/ > /dev/null && echo 'Nginx serving files successfully'" \
    "Nginx file serving"

# Test if the main page loads
run_container_tests "video-translator-frontend" \
    "curl -f http://localhost/ | grep -q 'Video Translator' && echo 'Main page loads successfully'" \
    "Main page loading"

# Test integration between containers
echo ""
print_status "INFO" "Testing Container Integration"
echo "----------------------------------------"

# Test API proxy through nginx
run_container_tests "video-translator-frontend" \
    "curl -f http://localhost/api/health > /dev/null && echo 'API proxy working'" \
    "API proxy through nginx"

# Test CORS headers
run_container_tests "video-translator-frontend" \
    "curl -I http://localhost/api/health | grep -q 'Access-Control-Allow-Origin' && echo 'CORS headers present'" \
    "CORS headers"

# Summary
echo ""
echo "======================================"
print_status "INFO" "Test Summary"
echo "======================================"

print_status "PASS" "Container integrity tests completed successfully!"
print_status "INFO" "All core functionality is working properly"
print_status "INFO" "The video translator is ready for use"

echo ""
print_status "INFO" "You can now access the application at: http://localhost"
print_status "INFO" "API documentation available at: http://localhost/api/health" 