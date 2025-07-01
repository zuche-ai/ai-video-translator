#!/bin/bash

# Video Translator Docker Runner
echo "🚀 Starting Video Translator with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Create directories if they don't exist
mkdir -p uploads results

# Build and start services
echo "📦 Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "✅ Video Translator is running!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:5001"
echo "📁 Uploads: ./uploads"
echo "📁 Results: ./results"
echo ""
echo "To stop: docker-compose down"
echo "To view logs: docker-compose logs -f" 