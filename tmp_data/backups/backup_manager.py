#!/usr/bin/env python3
"""
Employee Management System Backup Script
Automated PostgreSQL database backup with logging and email notifications
"""

import os
import subprocess
import datetime
import logging
import glob
from pathlib import Path

# Configuration
BACKUP_DIR = "/app/data/backups"  # Docker container path
DB_NAME = "employee_db"
DB_USER = "postgres"
RETENTION_DAYS = 30  # Keep backups for 30 days
LOG_FILE = "/app/logs/backup.log"  # Docker container path

# Weekly backup configuration
WEEKLY_BACKUP = True  # Set to True for weekly backups

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_backup():
    """Create PostgreSQL database backup"""
    try:
        # Create backup directory
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{BACKUP_DIR}/employee_db_backup_{timestamp}.sql"
        
        logger.info(f"Starting backup: {backup_file}")
        
        # Set environment variables for PostgreSQL connection
        env = os.environ.copy()
        env['PGPASSWORD'] = 'postgres'  # Set password for pg_dump
        
        # Create database dump - connect directly to postgres service via Docker network
        cmd = [
            "pg_dump",
            "-h", "postgres",  # hostname is the service name in docker-compose
            "-p", "5432",      # port
            "-U", DB_USER,     # username
            "-d", DB_NAME,     # database name
            "--clean", 
            "--if-exists",
            "--verbose"
        ]
        
        with open(backup_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, env=env)
        
        if result.returncode == 0:
            logger.info(f"Database dump successful: {backup_file}")
            
            # Compress backup
            compress_cmd = ["gzip", backup_file]
            subprocess.run(compress_cmd, check=True)
            compressed_file = f"{backup_file}.gz"
            
            logger.info(f"Backup compressed: {compressed_file}")
            
            # Get file size
            file_size = os.path.getsize(compressed_file)
            logger.info(f"Backup size: {file_size / (1024*1024):.2f} MB")
            
            return compressed_file
        else:
            logger.error(f"Database dump failed: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Backup creation failed: {str(e)}")
        return None

def cleanup_old_backups():
    """Remove old backup files"""
    try:
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=RETENTION_DAYS)
        pattern = f"{BACKUP_DIR}/employee_db_backup_*.sql.gz"
        
        deleted_count = 0
        for backup_file in glob.glob(pattern):
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(backup_file))
            if file_time < cutoff_date:
                os.remove(backup_file)
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup_file}")
        
        logger.info(f"Cleanup completed. Deleted {deleted_count} old backup(s)")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

def verify_backup(backup_file):
    """Verify backup file integrity"""
    try:
        if not os.path.exists(backup_file):
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        # Check if file is not empty
        file_size = os.path.getsize(backup_file)
        if file_size == 0:
            logger.error(f"Backup file is empty: {backup_file}")
            return False
        
        # Test gzip integrity
        test_cmd = ["gunzip", "-t", backup_file]
        result = subprocess.run(test_cmd, capture_output=True)
        
        if result.returncode == 0:
            logger.info(f"Backup verification successful: {backup_file}")
            return True
        else:
            logger.error(f"Backup verification failed: {backup_file}")
            return False
            
    except Exception as e:
        logger.error(f"Backup verification error: {str(e)}")
        return False

def main():
    """Main backup process"""
    logger.info("=" * 50)
    logger.info("Starting Employee Management System Backup")
    logger.info("=" * 50)
    
    # Create backup
    backup_file = create_backup()
    
    if backup_file:
        # Verify backup
        if verify_backup(backup_file):
            logger.info("Backup process completed successfully")
        else:
            logger.error("Backup verification failed")
            return 1
        
        # Cleanup old backups
        cleanup_old_backups()
        
        logger.info("Backup process finished")
        return 0
    else:
        logger.error("Backup process failed")
        return 1

if __name__ == "__main__":
    exit(main())
