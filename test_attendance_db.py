#!/usr/bin/env python3

import pandas as pd
import psycopg2
from datetime import datetime
import logging
import os
import sys

# Add the current directory to the path so we can import from core
sys.path.append('/Users/hriteekroy1869/Desktop/employee_management-main 2')

from core.data_seeder import DatabaseSeeder2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_attendance_seeding():
    """Test attendance data seeding with real database connection"""
    
    # Database configuration from docker-compose
    db_config = {
        'host': 'localhost',
        'port': '5435',
        'database': 'employee_db',
        'user': 'postgres',
        'password': 'postgres123'
    }
    
    # Check if CSV file exists
    csv_file = 'updated_csv_files/attendance-report.csv'
    if not os.path.exists(csv_file):
        logger.error(f"CSV file {csv_file} not found")
        return False
    
    try:
        # Test database connection first
        try:
            conn = psycopg2.connect(**db_config)
            conn.close()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.info("Make sure Docker containers are running with: docker compose up -d")
            return False
        
        # Load attendance data
        df_attendance = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df_attendance)} attendance records")
        
        # Initialize seeder
        seeder = DatabaseSeeder2(db_config)
        seeder.connect()
        
        # Check if attendance table exists and its structure
        seeder.cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'attendance'
            ORDER BY ordinal_position
        """)
        columns = seeder.cursor.fetchall()
        logger.info(f"Attendance table columns: {columns}")
        
        # Test the seeding method
        logger.info("Testing attendance seeding...")
        seeder.seed_attendance(df_attendance)
        
        # Check what was inserted
        seeder.cursor.execute("SELECT COUNT(*) FROM attendance")
        count = seeder.cursor.fetchone()[0]
        logger.info(f"Total attendance records in database: {count}")
        
        # Show a sample of inserted data
        seeder.cursor.execute("""
            SELECT attendance_date, employee_code, clock_in_time, clock_out_time, attendance_type
            FROM attendance 
            ORDER BY attendance_date DESC, employee_code
            LIMIT 5
        """)
        sample_data = seeder.cursor.fetchall()
        logger.info("Sample attendance data from database:")
        for record in sample_data:
            logger.info(f"  {record}")
        
        seeder.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Error testing attendance seeding: {e}")
        return False

if __name__ == "__main__":
    test_attendance_seeding()
