# Production Deployment Guide

This guide covers deploying the Video Translator application in a production environment with security, monitoring, and scalability considerations.

## Prerequisites

- Docker and Docker Compose installed
- Domain name (for SSL certificates)
- SSL certificates (Let's Encrypt recommended)
- Monitoring solution (optional but recommended)

## Quick Start

1. **Clone and navigate to the project:**
   ```bash
   cd video_translator
   ```

2. **Configure environment variables:**
   ```bash
   cp docker/production.env docker/.env
   # Edit docker/.env with your production values
   ```

3. **Start production services:**
   ```bash
   ./docker/run-prod.sh
   ```

## Security Configuration

### 1. Environment Variables

Update `docker/production.env` with secure values:

```bash
# Generate a secure secret key
SECRET_KEY=$(openssl rand -hex 32)

# Set your domain
DOMAIN=your-domain.com

# Configure rate limits
API_RATE_LIMIT=10
API_RATE_LIMIT_BURST=20
```

### 2. SSL/TLS Configuration

For production, enable HTTPS by uncommenting the SSL section in `docker/nginx.prod.conf`:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    # ... rest of SSL configuration
}
```

### 3. Firewall Configuration

Configure your firewall to only allow necessary ports:

```bash
# Allow HTTP, HTTPS, and SSH only
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

## Monitoring and Logging

### 1. Application Logs

Logs are stored in Docker volumes:
- Backend logs: `logs` volume
- Nginx logs: Inside frontend container

View logs:
```bash
# All services
docker-compose -f docker/docker-compose.prod.yml logs -f

# Specific service
docker-compose -f docker/docker-compose.prod.yml logs -f backend
```

### 2. Health Checks

The application includes health check endpoints:
- Frontend: `http://your-domain.com/health`
- Backend: `http://your-domain.com/api/health`

### 3. Resource Monitoring

Monitor container resources:
```bash
docker stats
```

## Backup Strategy

### 1. Data Backup

Backup important volumes:
```bash
# Backup uploads and results
docker run --rm -v video_translator_uploads:/data -v $(pwd):/backup alpine tar czf /backup/uploads-backup.tar.gz -C /data .
docker run --rm -v video_translator_results:/data -v $(pwd):/backup alpine tar czf /backup/results-backup.tar.gz -C /data .
```

### 2. Configuration Backup

Backup configuration files:
```bash
tar czf config-backup.tar.gz docker/
```

## Scaling Considerations

### 1. Horizontal Scaling

For high traffic, consider:
- Load balancer (HAProxy, Nginx)
- Multiple backend instances
- Database for job queue (Redis, PostgreSQL)

### 2. Resource Limits

Adjust resource limits in `docker-compose.prod.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Increase for heavy processing
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

## Maintenance

### 1. Updates

Update the application:
```bash
# Pull latest changes
git pull

# Rebuild and restart
./docker/run-prod.sh
```

### 2. Cleanup

Clean up old files:
```bash
# Remove old containers and images
docker system prune -f

# Clean up old uploads (older than 30 days)
find uploads/ -type f -mtime +30 -delete
```

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using port 80
   sudo lsof -i :80
   ```

2. **Memory issues:**
   ```bash
   # Check container memory usage
   docker stats
   ```

3. **Permission issues:**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER uploads/ results/ logs/
   ```

### Debug Mode

For debugging, use development compose:
```bash
docker-compose -f docker/docker-compose.yml up
```

## Performance Optimization

### 1. Nginx Optimization

Enable gzip compression and caching in `nginx.prod.conf` (already configured).

### 2. Application Optimization

- Use connection pooling for external APIs
- Implement caching for frequently accessed data
- Optimize video processing parameters

### 3. System Optimization

- Use SSD storage for better I/O performance
- Ensure sufficient RAM for video processing
- Monitor CPU usage during processing

## Security Checklist

- [ ] Changed default secret key
- [ ] Enabled HTTPS with valid certificates
- [ ] Configured firewall rules
- [ ] Set up rate limiting
- [ ] Enabled security headers
- [ ] Configured non-root users in containers
- [ ] Set up log monitoring
- [ ] Implemented backup strategy
- [ ] Regular security updates
- [ ] Access control and authentication (if needed)

## Support

For issues and questions:
1. Check the logs: `docker-compose -f docker/docker-compose.prod.yml logs`
2. Verify health checks: `curl http://your-domain.com/health`
3. Check resource usage: `docker stats`
4. Review this documentation

## Next Steps

Consider implementing:
- User authentication and authorization
- Database for persistent storage
- Job queue for background processing
- CDN for static assets
- Automated backups
- CI/CD pipeline
- Monitoring dashboard (Grafana, Prometheus) 