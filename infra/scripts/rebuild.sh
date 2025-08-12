#!/bin/bash
# Script to rebuild and restart the application without sudo

echo "===== STOPPING RUNNING CONTAINERS ====="
docker stop employee_manager_app employee_manager_db employee_backup_scheduler 2>/dev/null || true
docker rm employee_manager_app employee_manager_db employee_backup_scheduler 2>/dev/null || true

echo "===== REBUILDING DOCKER IMAGE ====="
cd /home/yamini/employee_management
docker build -t employee_management:latest .

echo "===== CREATING NETWORK (IF NOT EXISTS) ====="
docker network create app-network 2>/dev/null || true

echo "===== STARTING DATABASE CONTAINER ====="
docker run --name employee_manager_db -d \
  -e POSTGRES_DB=employee_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres123 \
  -e POSTGRES_HOST_AUTH_METHOD=trust \
  -p 5434:5432 \
  -v "$(pwd)/data:/var/lib/postgresql/data" \
  -v "$(pwd)/data:/app/data" \
  --network app-network \
  postgres:15-alpine

# Wait for database to be ready
echo "===== WAITING FOR DATABASE TO BE READY ====="
sleep 15

echo "===== STARTING APPLICATION CONTAINER ====="
docker run --name employee_manager_app -d \
  -p 8503:8501 \
  -v "$(pwd):/app" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/logs:/app/logs" \
  -e DATABASE_URL=postgresql://postgres:postgres123@employee_manager_db:5432/employee_db \
  --network app-network \
  employee_management:latest

echo "===== STARTING BACKUP SCHEDULER CONTAINER ====="
docker run --name employee_backup_scheduler -d \
  -v "$(pwd):/app" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/logs:/app/logs" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --network app-network \
  alpine:latest \
  sh -c "apk add --no-cache docker-cli dcron postgresql15-client python3 && \
        echo '0 2 * * 0 cd /app && sh docker-backup.sh' > /etc/crontabs/root && \
        echo '0 3 1 * * cd /app && python3 scripts/purge_logs.py 30' >> /etc/crontabs/root && \
        echo 'Starting scheduler services...' && \
        crond -f -l 2"

echo "===== APPLICATION REBUILT AND RESTARTED ====="
echo "Check application logs using:"
echo "docker logs employee_manager_app"
echo ""
echo "Access the application at: http://localhost:8503"
