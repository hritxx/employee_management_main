"""
ETL Pipeline for HR Management System
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg2.extras import execute_values

from config.config import etl_config, app_config, FILE_SCHEMAS, DATA_TYPE_RULES, db_config
from core.database import db_pool
from core.models import create_tables
from core.data_seeder import DatabaseSeeder2

logger = logging.getLogger(__name__)

class ETLPipeline:
    """Coordinates the ETL process"""

    def __init__(self):
        self.seeder = DatabaseSeeder2({
            'host': db_config.host,
            'database': db_config.database,
            'user': db_config.user,
            'password': db_config.password,
            'port': db_config.port
        })
        self.upload_id: Optional[int] = None

    def preprocess_allocations_csv(self, df):
        """Map uploaded allocations CSV columns to expected schema"""
        # First, get employee codes from employee master CSV
        try:
            emp_df = pd.read_csv('updated_csv_files/employee_master.csv')
            employee_map = dict(zip(emp_df['Employee Name'], emp_df['Employee Code']))
        except Exception as e:
            logger.error(f"Failed to read employee master data: {e}")
            raise
        
        # Map names to codes
        df['employee_code'] = df['Name'].map(employee_map)
        
        # Verify all names were mapped
        unmapped = df[df['employee_code'].isna()]['Name'].unique()
        if len(unmapped) > 0:
            logger.warning(f"Could not map employee codes for: {unmapped}")
        
        df = df.rename(columns={
            'Project Type': 'project_type',
            'Project Code': 'project_id',
            'Project Name': 'project_name',
            '% Allocation': 'allocation_percentage',
            'Role': 'role',
            'Idle Time (%)': 'idle_time_percentage',
            'Idle Time - Period': 'idle_time_period',
            'Available From': 'effective_from',
            'Comments': 'change_reason'
        })
        # Fill missing columns with default values
        for col in ['effective_to', 'status', 'created_by']:
            if col not in df.columns:
                df[col] = None
        df['status'] = 'active'
        df['created_by'] = 'system'
        # Ensure allocation_percentage is float
        if 'allocation_percentage' in df.columns:
            df['allocation_percentage'] = df['allocation_percentage'].astype(float)
        # Ensure effective_from is date
        if 'effective_from' in df.columns:
            df['effective_from'] = pd.to_datetime(df['effective_from']).dt.date
        return df

    def preprocess_timesheet_csv(self, df):
        """Map uploaded timesheet CSV columns to expected schema"""
        df = df.rename(columns={
            'Project': 'project_id',
            'Task': 'task_description',
            'Contributor': 'employee_code',
            'Date': 'work_date',
            'Duration in minutes': 'hours_worked'
        })
        # Convert minutes to hours
        if 'hours_worked' in df.columns:
            df['hours_worked'] = df['hours_worked'].astype(float) / 60.0
        return df

    def process_files(self, files: Dict[str, Path]) -> Tuple[bool, str, Dict]:
        """Process uploaded files through the ETL pipeline"""
        try:
            # Connect to database
            self.seeder.connect()

            # Log upload start
            logger.info(f"Starting ETL process with files: {[f.name for f in files.values()]}")
            
            # Read all CSV files
            df_dict = {}
            for file_type, file_path in files.items():
                logger.info(f"Reading {file_type} from {file_path}")
                df_dict[file_type] = pd.read_csv(file_path)
                logger.info(f"Successfully read {len(df_dict[file_type])} rows from {file_path}")

            # Preprocess allocations CSV if present
            if 'project_allocations' in df_dict:
                df_dict['project_allocations'] = self.preprocess_allocations_csv(df_dict['project_allocations'])
            # Preprocess timesheet CSV if present
            if 'timesheet_report' in df_dict:
                df_dict['timesheet_report'] = self.preprocess_timesheet_csv(df_dict['timesheet_report'])

            # Use DatabaseSeeder to load data
            success = self.seeder.seed_database(
                {
                    'employee_master': df_dict.get('employee_master'),
                    'employee_exit': df_dict.get('employee_exit'),
                    'experience_report': df_dict.get('experience_report'),
                    'work_profile': df_dict.get('work_profile'),
                    'attendance_report': df_dict.get('attendance_report'),
                    'timesheet_report': df_dict.get('timesheet_report'),
                    'project_allocations': df_dict.get('project_allocations'),
                    'resource_utilization': df_dict.get('resource_utilization')
                },
                clean_existing=False  # Don't clean existing data
            )

            # Close database connection
            self.seeder.disconnect()

            if success:
                return True, "Data loaded successfully", {
                    'stage': 'complete',
                    'extracted_files': len(df_dict),
                    'validation_errors': {}
                }
            else:
                return False, "Data loading failed", {
                    'stage': 'load',
                    'validation_errors': {}
                }

        except Exception as e:
            error_message = f"Pipeline execution failed: {str(e)}"
            logger.error(error_message)
            return False, error_message, {'stage': 'pipeline_error'}