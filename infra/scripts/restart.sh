#!/bin/bash
# Simple script to restart the application without sudo

echo "Stopping containers..."
docker stop employee_manager_app employee_manager_db employee_backup_scheduler 2>/dev/null || true

echo "Starting containers..."
cd /home/yamini/employee_management
docker run --name employee_manager_db -d \
  -e POSTGRES_DB=employee_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres123 \
  -e POSTGRES_HOST_AUTH_METHOD=trust \
  -p 5434:5432 \
  -v "$(pwd)/data:/app/data" \
  --network app-network \
  postgres:15-alpine

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 10

docker run --name employee_manager_app -d \
  -p 8503:8501 \
  -v "$(pwd):/app" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/logs:/app/logs" \
  -e DATABASE_URL=postgresql://postgres:postgres123@employee_manager_db:5432/employee_db \
  --network app-network \
  --link employee_manager_db \
  employee_management:latest

echo "Application restarted. Check logs using:"
echo "docker logs employee_manager_app"
