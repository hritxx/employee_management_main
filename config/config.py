"""
Configuration settings for the HR Management System
"""

import os
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
import logging

DATABASE_URL = os.getenv('DATABASE_URL')
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class DatabaseConfig:
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'employee_management_db')
    user: str = os.getenv('DB_USER', 'hr_admin')
    password: str = os.getenv('DB_PASSWORD', 'Aganitha@123')

@dataclass
class ETLConfig:
    batch_size: int = 1000
    max_workers: int = 4
    timeout_seconds: int = 300
    required_files: Dict[str, str] = None

    def __post_init__(self):
        self.required_files = {
            'employee_master': 'employee_master.csv',
            'employee_exit': 'employee_exit_report.csv',
            'experience_report': 'experience_report.csv',
            'work_profile': 'employee_work_profile.csv',
            'attendance_report': 'attendance_report_daily.csv',
            'timesheet_report': 'timesheet_report_clean.csv',
            'project_allocations': 'project_allocations.csv',
            'resource_utilization': 'resource_utilization.csv'
        }

@dataclass
class AppConfig:
    title: str = "HR Management System"
    description: str = "Enterprise ETL Pipeline for HR Data Management"
    version: str = "1.0.0"
    debug: bool = bool(os.getenv('APP_DEBUG', True))
    upload_folder: Path = Path("uploads")
    log_folder: Path = Path("logs")

    def __post_init__(self):
        self.upload_folder.mkdir(exist_ok=True)
        self.log_folder.mkdir(exist_ok=True)

# Create global config instances
db_config = DatabaseConfig()
etl_config = ETLConfig()
app_config = AppConfig()

# Validation schemas for each file type (updated to match actual formats)
FILE_SCHEMAS = {
    'employee_master': {
        'required_columns': [
            'Employee Code', 'Employee Name', 'Date Of Joining', 'Employee Type'
        ],
        'optional_columns': [
            'Email', 'Additional Email', 'Mobile Number', 'Secondary Mobile Number',
            'Gender', 'Date Of Birth', 'Marital Status', 'Office Location',
            'Business Unit', 'Designation', 'Department', 'Grade', 'Parent Department',
            'Primary Manager', 'Primary Manager Email', 'Bank Name', 'Branch Name',
            'Account Holder Name', 'Account Number', 'Account Type', 'IFSC Code',
            'PAN Number', 'Aadhaar Enrollment Number', 'Aadhaar Number',
            'Present Address', 'Present State', 'Present City', 'Present Pincode',
            'Present Country', 'Permanent Address', 'Permanent State', 'Permanent City',
            'Permanent Pincode', 'Permanent Country', 'Status'
        ]
    },
    'employee_exit': {
        'required_columns': [
            'Employee Code', 'Exit Date'
        ],
        'optional_columns': [
            'Employee Name', 'Department', 'Designation', 'Date Of Joining',
            'Notice Date', 'Exit Policy Name', 'Expected Resignation Date',
            'Notice Period Days', 'Reason', 'Reason Type', 'Reason of Exit',
            'Resigned Added On'
        ]
    },
    'experience_report': {
        'required_columns': [
            'Employee Code', 'Employee Name', 'Date Of Joining'
        ],
        'optional_columns': [
            'Business Unit', 'Department', 'Designation', 'Current Experience',
            'Past Experience', 'Total Experience'
        ]
    },
    'work_profile': {
        'required_columns': [
            'Employee Code', 'Employee Name'
        ],
        'optional_columns': [
            'Business Unit', 'Parent Department', 'Department', 'Designation',
            'Office Location'
        ]
    },
    'attendance_report': {
        'required_columns': [
            'Employee Code', 'Employee Name', 'ShiftDate'
        ],
        'optional_columns': [
            'Business Unit', 'Team', 'Department', 'Designation', 'DOJ',
            'Shift Name', 'Day', 'In Time', 'Out Time', 'Work Duration',
            'Break Duration', 'Late By', 'Over Time', 'Status'
        ]
    },
    'timesheet_report': {
        'required_columns': [
            'timesheet_id', 'work_date', 'employee_code', 'project_id', 'hours_worked'
        ],
        'optional_columns': [
            'task_description', 'allocation_id'
        ]
    },
    'project_allocations': {
        'required_columns': [
            'allocation_id', 'employee_code', 'project_id', 'allocation_percentage',
            'effective_from', 'effective_to', 'status'
        ],
        'optional_columns': [
            'created_by', 'change_reason'
        ]
    },
    'resource_utilization': {
        'required_columns': [
            'project_id','week_start_date', 'working_hours',
        ],
        'optional_columns': [
            'week_end_date'
        ]
    }
}

# Column mapping for actual CSV fields to database fields
COLUMN_MAPPINGS = {
    'employee_master': {
        'Employee Code': 'employee_code',
        'Employee Name': 'employee_name',
        'Email': 'email',
        'Mobile Number': 'mobile_number',
        'Date Of Joining': 'date_of_joining',
        'Employee Type': 'employee_type',
        'Gender': 'gender',
        'Date Of Birth': 'date_of_birth',
        'Marital Status': 'marital_status',
        'Business Unit': 'business_unit',
        'Department': 'department_name',
        'Designation': 'designation_name',
        'Grade': 'grade',
        'Parent Department': 'parent_department',
        'Status': 'status',
        'Present Address': 'present_address',
        'Permanent Address': 'permanent_address',
        'PAN Number': 'pan_number',
        'Aadhaar Number': 'aadhaar_number',
        'Bank Name': 'bank_name',
        'Account Number': 'account_number',
        'IFSC Code': 'ifsc_code'
    },
    'attendance_report': {
        'Employee Code': 'employee_code',
        'Employee Name': 'employee_name',
        'ShiftDate': 'attendance_date',
        'In Time': 'clock_in_time',
        'Out Time': 'clock_out_time',
        'Status': 'attendance_type'
    },
    'employee_exit': {
        'Employee Code': 'employee_code',
        'Employee Name': 'employee_name',
        'Exit Date': 'exit_date',
        'Expected Resignation Date': 'last_working_date',
        'Reason': 'exit_reason',
        'Reason of Exit': 'exit_comments'
    },
    'work_profile': {
        'Employee Code': 'employee_code',
        'Employee Name': 'employee_name',
        'Business Unit': 'business_unit',
        'Department': 'assigned_department',
        'Designation': 'designation',
        'Office Location': 'office_location_name'
    },
    'experience_report': {
        'Employee Code': 'employee_code',
        'Employee Name': 'employee_name',
        'Business Unit': 'business_unit',
        'Department': 'department',
        'Designation': 'designation',
        'Date Of Joining': 'date_of_joining',
        'Current Experience': 'current_experience',
        'Past Experience': 'past_experience',
        'Total Experience': 'total_experience'
    }
}

# Data type validation rules
DATA_TYPE_RULES = {
    'Employee Code': str,
    'Email': str,
    'Date Of Joining': 'date',
    'Date Of Birth': 'date',
    'Exit Date': 'date',
    'Expected Resignation Date': 'date',
    'Last Working Date': 'date',
    'Start Date': 'date',
    'End Date': 'date',
    'Date': 'date',
    'Total Experience': float,
    'Relevant Experience': float,
    'Allocation Percentage': float,
    'Hours Worked': float,
    'Billable Hours': float,
    'Working Hours': float,
    'Overtime': float
}