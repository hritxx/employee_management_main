#!/bin/bash
# Script to start the application with the full backup and logging system

echo "===== STARTING EMPLOYEE MANAGEMENT SYSTEM ====="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or you don't have permission to use it."
  echo "Please start Docker or add your user to the docker group."
  exit 1
fi

# Check if network exists, create if not
if ! docker network inspect app-network > /dev/null 2>&1; then
  echo "Creating Docker network: app-network"
  docker network create app-network
fi

# Check if containers are already running
if docker ps | grep -q "employee_manager"; then
  echo "Some containers are already running. Would you like to restart them? (y/n)"
  read answer
  if [[ "$answer" == "y" ]]; then
    echo "Stopping running containers..."
    docker stop employee_manager_app employee_manager_db employee_backup_scheduler 2>/dev/null || true
  else
    echo "Exiting without changes."
    exit 0
  fi
fi

echo "Starting the application with docker-compose..."
docker-compose -f docker-compose-with-backup.yml up -d

echo "===== STARTUP COMPLETE ====="
echo "The application is accessible at: http://localhost:8503"
echo ""
echo "To view logs:"
echo "  docker logs employee_manager_app"
echo ""
echo "To stop the application:"
echo "  docker-compose -f docker-compose-with-backup.yml down"
