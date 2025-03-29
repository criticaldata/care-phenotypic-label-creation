# Care Phenotype Analyzer Deployment Guide

This guide provides instructions for deploying the Care Phenotype Analyzer in various environments, from local development to production servers.

## Deployment Options

### 1. Local Development Deployment

#### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- Required system libraries

#### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/MIT-LCP/care-phenotypic-label-creation.git
cd care-phenotypic-label-creation
```

2. Create and activate virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n care-phenotype python=3.8
conda activate care-phenotype
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

### 2. Production Server Deployment

#### Prerequisites
- Linux/Unix server (Ubuntu 20.04 LTS recommended)
- Python 3.8 or higher
- Git
- System monitoring tools
- Backup solution

#### Server Setup

1. System requirements:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv git build-essential
```

2. Create deployment directory:
```bash
sudo mkdir -p /opt/care-phenotype
sudo chown -R $USER:$USER /opt/care-phenotype
```

3. Clone repository:
```bash
cd /opt/care-phenotype
git clone https://github.com/MIT-LCP/care-phenotypic-label-creation.git .
```

4. Set up Python environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

5. Configure systemd service:
```bash
sudo nano /etc/systemd/system/care-phenotype.service
```

Add the following content:
```ini
[Unit]
Description=Care Phenotype Analyzer Service
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/opt/care-phenotype
Environment="PATH=/opt/care-phenotype/venv/bin"
ExecStart=/opt/care-phenotype/venv/bin/python -m care_phenotype_analyzer
Restart=always

[Install]
WantedBy=multi-user.target
```

6. Start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable care-phenotype
sudo systemctl start care-phenotype
```

### 3. Docker Deployment

#### Prerequisites
- Docker
- Docker Compose

#### Docker Setup

1. Build the Docker image:
```bash
docker build -t care-phenotype-analyzer .
```

2. Run the container:
```bash
docker run -d \
  --name care-phenotype \
  -p 8000:8000 \
  -v /path/to/data:/data \
  care-phenotype-analyzer
```

3. Using Docker Compose:
```yaml
version: '3.8'
services:
  care-phenotype:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - /path/to/data:/data
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the deployment directory:

```env
# Application Settings
DEBUG=False
LOG_LEVEL=INFO
DATA_DIR=/data

# Security
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=your-domain.com

# Database (if using)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=care_phenotype
DB_USER=your_user
DB_PASSWORD=your_password

# Monitoring
ENABLE_MONITORING=True
PROMETHEUS_PORT=9090
```

### 2. Logging Configuration

Configure logging in `logging_config.yaml`:

```yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/care-phenotype/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: standard
  console:
    class: logging.StreamHandler
    formatter: standard
loggers:
  care_phenotype_analyzer:
    level: INFO
    handlers: [file, console]
```

## Monitoring and Maintenance

### 1. Health Checks

Implement health check endpoint:
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.now().isoformat()
    }
```

### 2. Backup Strategy

1. Database backups (if using):
```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backup/care-phenotype"
DATE=$(date +%Y%m%d)
pg_dump -U your_user care_phenotype > $BACKUP_DIR/backup_$DATE.sql
```

2. Configuration backups:
```bash
# Backup configuration files
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /etc/care-phenotype/
```

### 3. Monitoring Setup

1. Install monitoring tools:
```bash
# Prometheus
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v /path/to/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

2. Configure monitoring:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'care-phenotype'
    static_configs:
      - targets: ['localhost:8000']
```

## Security Considerations

### 1. Access Control

1. Configure firewall:
```bash
sudo ufw allow 8000/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

2. Set up SSL/TLS:
```bash
# Using Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 2. Data Security

1. Encrypt sensitive data:
```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)
```

2. Secure file permissions:
```bash
sudo chown -R care-phenotype:care-phenotype /opt/care-phenotype
sudo chmod -R 750 /opt/care-phenotype
```

## Troubleshooting

### 1. Common Issues

1. Service won't start:
```bash
# Check logs
sudo journalctl -u care-phenotype -n 50

# Check permissions
ls -la /opt/care-phenotype
```

2. Performance issues:
```bash
# Monitor system resources
top
htop
```

### 2. Recovery Procedures

1. Service recovery:
```bash
# Restart service
sudo systemctl restart care-phenotype

# Check status
sudo systemctl status care-phenotype
```

2. Data recovery:
```bash
# Restore from backup
pg_restore -U your_user -d care_phenotype backup_20240101.sql
```

## Scaling Considerations

### 1. Horizontal Scaling

1. Load balancer setup:
```nginx
# nginx.conf
upstream care_phenotype {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://care_phenotype;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

2. Multiple instances:
```bash
# Start multiple instances
python -m care_phenotype_analyzer --port 8001 &
python -m care_phenotype_analyzer --port 8002 &
```

### 2. Resource Optimization

1. Memory management:
```python
# Configure memory limits
import resource
resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))  # 1GB limit
```

2. Process management:
```bash
# Use process manager
pip install supervisor
```

## Maintenance Procedures

### 1. Regular Maintenance

1. Update dependencies:
```bash
pip install --upgrade -r requirements.txt
```

2. Clean up logs:
```bash
find /var/log/care-phenotype -name "*.log" -mtime +30 -delete
```

### 2. Version Updates

1. Update process:
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

2. Rollback procedure:
```bash
# Revert to previous version
git checkout v1.0.0
docker-compose down
docker-compose up -d
``` 