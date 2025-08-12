#!/usr/bin/env python3
"""
Automated log purge script to clean up logs older than 30 days.
This can be run as a scheduled task (e.g., cron job) weekly.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to path to allow importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Load environment variables
load_dotenv()

from logs.activity_logger import get_logger

def purge_old_logs(days_to_keep=30):
    """Purge logs older than the specified number of days"""
    print(f"Starting log purge process at {datetime.now()}")
    
    try:
        # Get logger instance
        logger = get_logger()
        
        # Log the purge operation start
        logger.log_event(
            event_type="SYSTEM_MAINTENANCE",
            description=f"Starting log purge for records older than {days_to_keep} days",
            user="system",
            details={
                "operation": "log_purge",
                "retention_days": days_to_keep,
                "timestamp": str(datetime.now())
            }
        )
        
        # Perform the purge
        deleted_count = logger.purge_old_logs(days_to_keep)
        
        # Log the purge operation completion
        logger.log_event(
            event_type="SYSTEM_MAINTENANCE",
            description=f"Completed log purge: Removed {deleted_count} records older than {days_to_keep} days",
            user="system",
            details={
                "operation": "log_purge",
                "records_deleted": deleted_count,
                "retention_days": days_to_keep,
                "timestamp": str(datetime.now())
            }
        )
        
        print(f"Log purge complete: Removed {deleted_count} records")
        return True
        
    except Exception as e:
        print(f"Error during log purge: {str(e)}")
        return False

if __name__ == "__main__":
    # Default to 30 days, but allow command-line override
    days = 30
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print(f"Invalid days value: {sys.argv[1]}. Using default of 30 days.")
    
    success = purge_old_logs(days)
    sys.exit(0 if success else 1)
