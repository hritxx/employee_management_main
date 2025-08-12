#!/bin/bash

# Set environment variables to match Docker Compose configuration
export DB_HOST=localhost
export DB_PORT=5433
export DB_NAME=employee_db
export DB_USER=postgres
export DB_PASSWORD=postgres123

# Run the seed script
echo "Starting project data seeding with Docker database configuration..."
echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
echo "User: $DB_USER"

python seed_projects.py
