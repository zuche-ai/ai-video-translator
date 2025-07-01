# Docker Setup for Video Translator

This project now supports running in Docker containers for easy deployment and consistent environments.

## Prerequisites

- Docker Desktop installed and running
- At least 4GB of available RAM (for AI models)

## Quick Start

### Option 1: Using the Helper Script

```bash
# Make the script executable (first time only)
chmod +x docker-run.sh

# Start the services
./docker-run.sh
```

### Option 2: Manual Docker Compose

```bash
# Build and start services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Accessing the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:5001
- **Health Check**: http://localhost:5001/health

## File Storage

- **Uploads**: `./uploads/` (mounted as volume)
- **Results**: `./results/` (mounted as volume)

## Services

### Backend (Python/Flask)
- **Port**: 5001
- **Features**: Video processing, transcription, translation, voice cloning
- **Dependencies**: FFmpeg, Python packages, Rust compiler

### Frontend (React/nginx)
- **Port**: 3000
- **Features**: Drag-and-drop UI, progress tracking, batch processing
- **Proxy**: API requests to backend

## Management Commands

```bash
# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Rebuild after code changes
docker-compose build
docker-compose up -d

# Clean up (removes containers and images)
docker-compose down --rmi all --volumes --remove-orphans
```

## Development

### Rebuilding After Changes

```bash
# Rebuild specific service
docker-compose build backend
docker-compose build frontend

# Rebuild and restart
docker-compose up --build -d
```

### Debugging

```bash
# Access backend container
docker exec -it video-translator-backend bash

# Access frontend container
docker exec -it video-translator-frontend sh

# Check container logs
docker-compose logs backend
docker-compose logs frontend
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change ports in `docker-compose.yml`
2. **Out of memory**: Increase Docker Desktop memory allocation
3. **Build failures**: Check Docker logs for dependency issues
4. **Permission errors**: Ensure uploads/ and results/ directories exist

### Health Checks

```bash
# Test backend health
curl http://localhost:5001/health

# Test frontend
curl http://localhost:3000
```

## Production Considerations

- Use production WSGI server (gunicorn) instead of Flask development server
- Configure proper logging
- Set up reverse proxy (nginx) for SSL termination
- Use Docker secrets for sensitive configuration
- Implement proper backup strategy for volumes

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │
│   (React)       │    │   (Python)      │
│   Port: 3000    │◄──►│   Port: 5001    │
└─────────────────┘    └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   nginx         │    │   uploads/      │
│   (static)      │    │   results/      │
└─────────────────┘    └─────────────────┘
``` 