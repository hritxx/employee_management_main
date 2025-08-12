#!/bin/bash

# Weekly Database Backup Script for Docker
# This script runs inside a Docker container to perform weekly backups

set -e

# Configuration
BACKUP_DIR="/app/data/backups"
CONTAINER_NAME="employee_manager_db"
DB_NAME="employee_db"
DB_USER="postgres"
RETENTION_DAYS=30
LOG_FILE="/app/logs/backup.log"

# Create backup directory
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_message "Starting weekly database backup"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/employee_db_backup_$TIMESTAMP.sql"

log_message "Creating backup: $BACKUP_FILE"

# Create database dump
export PGPASSWORD="postgres123"
if pg_dump -h postgres -U "$DB_USER" -d "$DB_NAME" --clean --if-exists -f "$BACKUP_FILE"; then
    log_message "Database dump successful"
    
    # Compress backup
    if gzip "$BACKUP_FILE"; then
        COMPRESSED_FILE="$BACKUP_FILE.gz"
        FILE_SIZE=$(du -h "$COMPRESSED_FILE" | cut -f1)
        log_message "Backup compressed successfully: $COMPRESSED_FILE ($FILE_SIZE)"
        
        # Verify backup
        if gunzip -t "$COMPRESSED_FILE" 2>/dev/null; then
            log_message "Backup verification successful"
        else
            log_message "ERROR: Backup verification failed"
            exit 1
        fi
    else
        log_message "ERROR: Backup compression failed"
        exit 1
    fi
else
    log_message "ERROR: Database dump failed"
    exit 1
fi

# Cleanup old backups
log_message "Cleaning up old backups (older than $RETENTION_DAYS days)"
DELETED_COUNT=0

find "$BACKUP_DIR" -name "employee_db_backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -print0 | while IFS= read -r -d '' file; do
    if rm "$file"; then
        log_message "Deleted old backup: $file"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

log_message "Weekly backup completed successfully"
