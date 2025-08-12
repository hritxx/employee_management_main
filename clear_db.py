"""
Database clearing functionality for the HR Management System
"""

import os
import psycopg2
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Get DB credentials from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5434")
DB_NAME = os.getenv("DB_NAME", "employee_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123")

# List of tables to clear in order (respecting foreign key constraints)
TABLES = [
    "task_summary_history",
    "task_summary", 
    "data_validation_errors",
    "csv_upload_log",
    "employee_exit",
    "attendance",
    "timesheet",
    "project_allocation",
    "employee_personal",
    "employee_financial",
    "employee",
    "project",
    "department",
    "designation"
]

def clear_tables():
    """
    Clear all data from database tables while preserving table structure.
    Tables are cleared in order to respect foreign key constraints.
    """
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST, 
            port=DB_PORT, 
            dbname=DB_NAME, 
            user=DB_USER, 
            password=DB_PASSWORD
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # Disable foreign key checks temporarily
        cur.execute("SET session_replication_role = 'replica';")
        logger.info("Disabled foreign key constraints")
        
        # Clear each table
        cleared_tables = []
        for table in TABLES:
            try:
                cur.execute(f"DELETE FROM {table};")
                cleared_tables.append(table)
                logger.info(f"Cleared table: {table}")
            except psycopg2.Error as e:
                logger.warning(f"Could not clear table {table}: {e}")
                # Continue with other tables even if one fails
                continue
        
        # Re-enable foreign key checks
        cur.execute("SET session_replication_role = 'origin';")
        logger.info("Re-enabled foreign key constraints")
        
        # Close connections
        cur.close()
        conn.close()
        
        logger.info(f"Successfully cleared {len(cleared_tables)} tables")
        return True, f"Successfully cleared {len(cleared_tables)} tables: {', '.join(cleared_tables)}"
        
    except psycopg2.Error as e:
        error_msg = f"Database error while clearing tables: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error while clearing tables: {e}"
        logger.error(error_msg)
        return False, error_msg

if __name__ == "__main__":
    success, message = clear_tables()
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
