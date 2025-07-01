# Docker Setup for Video Translator

This project now supports running in Docker containers for easy deployment and consistent environments.

## Prerequisites

- Docker Desktop installed and running
- At least 4GB of available RAM (for AI models)

## Quick Start

### Option 1: Using the Helper Script

```bash
# Development mode
./docker/docker-run.sh

# Production mode
./docker/run-prod.sh
```

### Option 2: Manual Docker Compose

```bash
# Development mode
docker-compose -f docker/docker-compose.yml up --build -d

# Production mode
docker-compose -f docker/docker-compose.prod.yml up --build -d

# Check status
docker-compose -f docker/docker-compose.yml ps

# View logs
docker-compose -f docker/docker-compose.yml logs -f
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
# Development mode
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml restart
docker-compose -f docker/docker-compose.yml logs -f backend
docker-compose -f docker/docker-compose.yml logs -f frontend
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up -d

# Production mode
docker-compose -f docker/docker-compose.prod.yml down
docker-compose -f docker/docker-compose.prod.yml restart
docker-compose -f docker/docker-compose.prod.yml logs -f backend
docker-compose -f docker/docker-compose.prod.yml logs -f frontend
docker-compose -f docker/docker-compose.prod.yml build
docker-compose -f docker/docker-compose.prod.yml up -d

# Clean up (removes containers and images)
docker-compose -f docker/docker-compose.yml down --rmi all --volumes --remove-orphans
```

## Development

### Rebuilding After Changes

```bash
# Development mode
docker-compose -f docker/docker-compose.yml build backend
docker-compose -f docker/docker-compose.yml build frontend
docker-compose -f docker/docker-compose.yml up --build -d

# Production mode
docker-compose -f docker/docker-compose.prod.yml build backend
docker-compose -f docker/docker-compose.prod.yml build frontend
docker-compose -f docker/docker-compose.prod.yml up --build -d
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

## Docker Organization

The Docker files are organized in the `docker/` directory:

```
docker/
├── docker-compose.yml          # Development configuration
├── docker-compose.prod.yml     # Production configuration
├── Dockerfile                  # Development backend
├── Dockerfile.prod             # Production backend
├── Dockerfile.frontend.prod    # Production frontend
├── nginx.prod.conf             # Production nginx config
├── docker-run.sh               # Development startup script
├── run-prod.sh                 # Production startup script
├── production.env              # Production environment variables
└── PRODUCTION.md               # Production deployment guide
```

## Development vs Production

### Development Mode
- Uses Flask development server
- Exposes ports 3000 (frontend) and 5001 (backend)
- Includes development tools and debugging
- Suitable for local development and testing

### Production Mode
- Uses Gunicorn WSGI server
- Nginx reverse proxy with security headers
- Non-root users for security
- Health checks and monitoring
- Rate limiting and resource constraints
- Optimized for performance and security

For detailed production deployment instructions, see [docker/PRODUCTION.md](docker/PRODUCTION.md).

## Production Considerations

- Use production WSGI server (gunicorn) instead of Flask development server
- Configure proper logging
- Set up reverse proxy (nginx) for SSL termination
- Use Docker secrets for sensitive configuration
- Implement proper backup strategy for volumes
- Enable security headers and rate limiting
- Use non-root users in containers
- Set up monitoring and health checks

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