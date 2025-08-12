#!/usr/bin/env python3
"""
Database Table Creation Script
Creates HR management system tables in PostgreSQL or SQLite database
"""

import psycopg2
import sqlite3
import logging
from typing import Union, Optional
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'db_creation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatabaseTableCreator:
    """Creates tables for HR management system"""

    def __init__(self, db_type: str = "postgresql"):
        self.db_type = db_type.lower()
        self.connection = None

    def connect_postgresql(self, host: str, database: str, user: str, password: str, port: int = 5433):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
            logger.info(f"Connected to PostgreSQL database: {database}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def connect_sqlite(self, db_path: str = "hr_management.db"):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(db_path)
            logger.info(f"Connected to SQLite database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    def get_table_creation_queries(self):
        """Return table creation queries based on database type"""

        if self.db_type == "postgresql":
            return self._get_postgresql_queries()
        else:
            return self._get_sqlite_queries()

    def _get_postgresql_queries(self):
        """PostgreSQL table creation queries"""
        return [
            # DEPARTMENT
            """
            CREATE TABLE IF NOT EXISTS department (
                department_id SERIAL PRIMARY KEY,
                department_name VARCHAR(100) NOT NULL,
                business_unit VARCHAR(100) NOT NULL,
                parent_department VARCHAR(100),
                status VARCHAR(20) DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # DESIGNATION
            """
            CREATE TABLE IF NOT EXISTS designation (
                designation_id SERIAL PRIMARY KEY,
                designation_name VARCHAR(100) NOT NULL,
                level VARCHAR(50),
                status VARCHAR(20) DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # EMPLOYEE
            """
            CREATE TABLE IF NOT EXISTS employee (
                employee_code VARCHAR(20) PRIMARY KEY,
                employee_name VARCHAR(200) NOT NULL,
                email VARCHAR(255) UNIQUE,
                mobile_number VARCHAR(20),
                date_of_joining DATE NOT NULL,
                employee_type VARCHAR(50) NOT NULL,
                grade VARCHAR(20),
                status VARCHAR(20) DEFAULT 'Active',
                department_id INTEGER REFERENCES department(department_id),
                department_name VARCHAR(100),
                designation_id INTEGER REFERENCES designation(designation_id),
                primary_manager_id VARCHAR(20) REFERENCES employee(employee_code),
                past_experience DECIMAL(5,2) DEFAULT 0,
                current_experience DECIMAL(5,2) DEFAULT 0,
                total_experience DECIMAL(5,2) GENERATED ALWAYS AS (current_experience + past_experience) STORED,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # EMPLOYEE_PERSONAL
            """
            CREATE TABLE IF NOT EXISTS employee_personal (
                employee_code VARCHAR(20) PRIMARY KEY REFERENCES employee(employee_code) ON DELETE CASCADE,
                gender VARCHAR(10) CHECK (gender IN ('Male','Female','Other')),
                date_of_birth DATE,
                marital_status VARCHAR(20),
                present_address TEXT,
                permanent_address TEXT,
                pan_number VARCHAR(20) UNIQUE,
                aadhaar_number VARCHAR(20) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # EMPLOYEE_FINANCIAL
            """
            CREATE TABLE IF NOT EXISTS employee_financial (
                employee_code VARCHAR(20) PRIMARY KEY REFERENCES employee(employee_code) ON DELETE CASCADE,
                bank_name VARCHAR(100),
                account_number VARCHAR(50) UNIQUE,
                ifsc_code VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # EMPLOYEE_WORK_PROFILE
            """
            CREATE TABLE IF NOT EXISTS employee_work_profile (
                employee_code VARCHAR(20) PRIMARY KEY REFERENCES employee(employee_code) ON DELETE CASCADE,
                role VARCHAR(100),
                primary_skills TEXT[],
                secondary_skills TEXT[],
                total_experience_years DECIMAL(5,2),
                relevant_experience_years DECIMAL(5,2),
                certifications TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # PROJECT
            """
            CREATE TABLE IF NOT EXISTS project (
                project_id VARCHAR(50) PRIMARY KEY,
                project_name VARCHAR(200) NOT NULL,
                client_name VARCHAR(200),
                status VARCHAR(20) DEFAULT 'Active',
                start_date DATE,
                end_date DATE,
                manager_id VARCHAR(20) REFERENCES employee(employee_code),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # PROJECT_ALLOCATION
            """
            CREATE TABLE IF NOT EXISTS project_allocation (
                allocation_id SERIAL PRIMARY KEY,
                employee_code VARCHAR(20) NOT NULL REFERENCES employee(employee_code),
                project_id VARCHAR(50) NOT NULL REFERENCES project(project_id),
                allocation_percentage DECIMAL(5,2) CHECK (allocation_percentage >= 0 AND allocation_percentage <= 100),
                effective_from DATE NOT NULL,
                effective_to DATE,
                status VARCHAR(20) DEFAULT 'Active',
                created_by VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                change_reason TEXT,
                CONSTRAINT chk_effective_dates CHECK (effective_to IS NULL OR effective_to >= effective_from)
            );
            """,

            # TIMESHEET
            """
            CREATE TABLE IF NOT EXISTS timesheet (
                timesheet_id SERIAL PRIMARY KEY,
                work_date DATE NOT NULL,
                employee_code VARCHAR(20) NOT NULL REFERENCES employee(employee_code),
                project_id VARCHAR(50) NOT NULL REFERENCES project(project_id),
                hours_worked DECIMAL(4,2) CHECK (hours_worked >= 0 AND hours_worked <= 24),
                task_description TEXT,
                allocation_id INTEGER REFERENCES project_allocation(allocation_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(employee_code, project_id, work_date)
            );
            """,

            # ATTENDANCE
            """
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id SERIAL PRIMARY KEY,
                attendance_date DATE NOT NULL,
                employee_code VARCHAR(20) NOT NULL REFERENCES employee(employee_code),
                clock_in_time TIME,
                clock_out_time TIME,
                total_hours DECIMAL(4,2) GENERATED ALWAYS AS (
                    CASE 
                        WHEN clock_in_time IS NOT NULL AND clock_out_time IS NOT NULL 
                        THEN EXTRACT(EPOCH FROM (clock_out_time - clock_in_time)) / 3600
                        ELSE NULL
                    END
                ) STORED,
                attendance_type VARCHAR(20) DEFAULT 'Present',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(employee_code, attendance_date)
            );
            """,

            # EMPLOYEE_EXIT
            """
            CREATE TABLE IF NOT EXISTS employee_exit (
                exit_id SERIAL PRIMARY KEY,
                employee_code VARCHAR(20) UNIQUE NOT NULL REFERENCES employee(employee_code),
                exit_date DATE NOT NULL,
                last_working_date DATE,
                exit_reason VARCHAR(200),
                exit_comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # CSV_UPLOAD_LOG
            """
            CREATE TABLE IF NOT EXISTS csv_upload_log (
                upload_id SERIAL PRIMARY KEY,
                file_name VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                uploaded_by VARCHAR(20) REFERENCES employee(employee_code),
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                records_processed INTEGER DEFAULT 0,
                records_success INTEGER DEFAULT 0,
                records_failed INTEGER DEFAULT 0,
                error_log TEXT,
                status VARCHAR(20) DEFAULT 'Processing'
            );
            """,

            # DATA_VALIDATION_ERRORS
            """
            CREATE TABLE IF NOT EXISTS data_validation_errors (
                error_id SERIAL PRIMARY KEY,
                upload_id INTEGER NOT NULL REFERENCES csv_upload_log(upload_id),
                row_number INTEGER,
                field_name VARCHAR(100),
                field_value TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # TASK_SUMMARY
            """
            CREATE TABLE IF NOT EXISTS task_summary (
                summary_id SERIAL PRIMARY KEY,
                employee_code VARCHAR(20) NOT NULL REFERENCES employee(employee_code),
                project_id VARCHAR(50) NOT NULL REFERENCES project(project_id),
                summary_text TEXT NOT NULL,
                summary_type VARCHAR(50) DEFAULT 'AI_GENERATED',
                model_used VARCHAR(100),
                total_hours NUMERIC(8,2),
                task_count INTEGER,
                date_range_start DATE,
                date_range_end DATE,
                generated_by VARCHAR(20),
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1,
                status VARCHAR(20) DEFAULT 'Active',
                metadata JSONB
            );
            """,

            # TASK_SUMMARY_HISTORY
            """
            CREATE TABLE IF NOT EXISTS task_summary_history (
                history_id SERIAL PRIMARY KEY,
                summary_id INTEGER NOT NULL REFERENCES task_summary(summary_id) ON DELETE CASCADE,
                old_summary_text TEXT,
                new_summary_text TEXT,
                old_version INTEGER,
                new_version INTEGER,
                change_reason TEXT,
                changed_by VARCHAR(20),
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # RESOURCE_UTILIZATION (commented out)
            # """
            # CREATE TABLE IF NOT EXISTS resource_utilization (
            # project_id VARCHAR(50) NOT NULL,
            # week_start_date DATE NOT NULL,
            # estimated_hours NUMERIC(10,2) NOT NULL DEFAULT 0,
            # PRIMARY KEY (project_id, week_start_date),
            # FOREIGN KEY (project_id) REFERENCES project(project_id)
            # );
            # """
        ]

    def _get_sqlite_queries(self):
        """SQLite table creation queries (adapted for SQLite syntax)"""
        return [
            # DEPARTMENT
            """
            CREATE TABLE IF NOT EXISTS department (
                department_id INTEGER PRIMARY KEY AUTOINCREMENT,
                department_name VARCHAR(100) NOT NULL,
                business_unit VARCHAR(100) NOT NULL,
                parent_department VARCHAR(100),
                status VARCHAR(20) DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # DESIGNATION
            """
            CREATE TABLE IF NOT EXISTS designation (
                designation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                designation_name VARCHAR(100) NOT NULL,
                level VARCHAR(50),
                status VARCHAR(20) DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # EMPLOYEE
            """
            CREATE TABLE IF NOT EXISTS employee (
                employee_code VARCHAR(20) PRIMARY KEY,
                employee_name VARCHAR(200) NOT NULL,
                email VARCHAR(255) UNIQUE,
                mobile_number VARCHAR(20),
                date_of_joining DATE NOT NULL,
                employee_type VARCHAR(50) NOT NULL,
                grade VARCHAR(20),
                status VARCHAR(20) DEFAULT 'Active',
                department_id INTEGER REFERENCES department(department_id),
                department_name VARCHAR(100),
                designation_id INTEGER REFERENCES designation(designation_id),
                primary_manager_id VARCHAR(20) REFERENCES employee(employee_code),
                past_experience DECIMAL(5,2) DEFAULT 0,
                current_experience DECIMAL(5,2) DEFAULT 0,
                total_experience DECIMAL(5,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # EMPLOYEE_PERSONAL
            """
            CREATE TABLE IF NOT EXISTS employee_personal (
                employee_code VARCHAR(20) PRIMARY KEY REFERENCES employee(employee_code) ON DELETE CASCADE,
                gender VARCHAR(10) CHECK (gender IN ('Male','Female','Other')),
                date_of_birth DATE,
                marital_status VARCHAR(20),
                present_address TEXT,
                permanent_address TEXT,
                pan_number VARCHAR(20) UNIQUE,
                aadhaar_number VARCHAR(20) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # EMPLOYEE_FINANCIAL (Note: SQLite doesn't support GENERATED columns in the same way)
            """
            CREATE TABLE IF NOT EXISTS employee_financial (
                employee_code VARCHAR(20) PRIMARY KEY REFERENCES employee(employee_code) ON DELETE CASCADE,
                bank_name VARCHAR(100),
                account_number VARCHAR(50) UNIQUE,
                ifsc_code VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # EMPLOYEE_WORK_PROFILE table before PROJECT table
            """
            CREATE TABLE IF NOT EXISTS employee_work_profile (
                employee_code VARCHAR(20) PRIMARY KEY REFERENCES employee(employee_code) ON DELETE CASCADE,
                role VARCHAR(100),
                primary_skills TEXT[],
                secondary_skills TEXT[],
                total_experience_years DECIMAL(5,2),
                relevant_experience_years DECIMAL(5,2),
                certifications TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # PROJECT
            """
            CREATE TABLE IF NOT EXISTS project (
                project_id VARCHAR(50) PRIMARY KEY,
                project_name VARCHAR(200) NOT NULL,
                client_name VARCHAR(200),
                status VARCHAR(20) DEFAULT 'Active',
                start_date DATE,
                end_date DATE,
                manager_id VARCHAR(20) REFERENCES employee(employee_code),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # PROJECT_ALLOCATION
            """
            CREATE TABLE IF NOT EXISTS project_allocation (
                allocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_name VARCHAR(200) NOT NULL REFERENCES employee(employee_name),
                project_id VARCHAR(50) NOT NULL REFERENCES project(project_id),
                allocation_percentage DECIMAL(5,2) CHECK (allocation_percentage >= 0 AND allocation_percentage <= 100),
                effective_from DATE NOT NULL,
                effective_to DATE,
                status VARCHAR(20) DEFAULT 'Active',
                created_by VARCHAR(20) REFERENCES employee(employee_code),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                change_reason TEXT,
                CHECK (effective_to IS NULL OR effective_to >= effective_from)
            );
            """,

            # TIMESHEET
            """
            CREATE TABLE IF NOT EXISTS timesheet (
                timesheet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_date DATE NOT NULL,
                employee_code VARCHAR(20) NOT NULL REFERENCES employee(employee_code),
                project_id VARCHAR(50) NOT NULL REFERENCES project(project_id),
                hours_worked DECIMAL(4,2) CHECK (hours_worked >= 0 AND hours_worked <= 24),
                task_description TEXT,
                allocation_id INTEGER REFERENCES project_allocation(allocation_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(employee_code, project_id, work_date)
            );
            """,

            # ATTENDANCE (SQLite doesn't support TIME calculations in generated columns)
            """
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                attendance_date DATE NOT NULL,
                employee_code VARCHAR(20) NOT NULL REFERENCES employee(employee_code),
                clock_in_time TIME,
                clock_out_time TIME,
                total_hours DECIMAL(4,2),
                attendance_type VARCHAR(20) DEFAULT 'Present',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(employee_code, attendance_date)
            );
            """,

            # EMPLOYEE_EXIT
            """
            CREATE TABLE IF NOT EXISTS employee_exit (
                exit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_code VARCHAR(20) UNIQUE NOT NULL REFERENCES employee(employee_code),
                exit_date DATE NOT NULL,
                last_working_date DATE,
                exit_reason VARCHAR(200),
                exit_comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # CSV_UPLOAD_LOG
            """
            CREATE TABLE IF NOT EXISTS csv_upload_log (
                upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                uploaded_by VARCHAR(20) REFERENCES employee(employee_code),
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                records_processed INTEGER DEFAULT 0,
                records_success INTEGER DEFAULT 0,
                records_failed INTEGER DEFAULT 0,
                error_log TEXT,
                status VARCHAR(20) DEFAULT 'Processing'
            );
            """,

            # DATA_VALIDATION_ERRORS
            """
            CREATE TABLE IF NOT EXISTS data_validation_errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER NOT NULL REFERENCES csv_upload_log(upload_id),
                row_number INTEGER,
                field_name VARCHAR(100),
                field_value TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
                        # RESOURCE_UTILIZATION
            """
            CREATE TABLE IF NOT EXISTS resource_utilization (
                utilization_id SERIAL PRIMARY KEY,
                employee_code VARCHAR(20) NOT NULL REFERENCES employee(employee_code),
                project_id VARCHAR(50) REFERENCES project(project_id),
                date_period DATE NOT NULL,
                period_type VARCHAR(20) DEFAULT 'monthly',
                allocated_hours DECIMAL(8,2) DEFAULT 0,
                actual_hours DECIMAL(8,2) DEFAULT 0,
                utilization_percentage DECIMAL(5,2) GENERATED ALWAYS AS (
                    CASE 
                        WHEN allocated_hours > 0 
                        THEN (actual_hours / allocated_hours) * 100
                        ELSE 0
                    END
                ) STORED,
                billable_hours DECIMAL(8,2) DEFAULT 0,
                non_billable_hours DECIMAL(8,2) DEFAULT 0,
                status VARCHAR(20) DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]

    def create_tables(self):
        """Create all tables in the database"""
        if not self.connection:
            raise Exception("No database connection established")

        queries = self.get_table_creation_queries()
        table_names = [
            'department', 'designation', 'employee', 'employee_personal',
            'employee_financial', 'employee_work_profile', 'project', 'project_allocation', 'timesheet',
            'attendance', 'employee_exit', 'csv_upload_log', 'data_validation_errors',
            'task_summary', 'task_summary_history'
            # 'resource_utilization'  # keep commented if table is commented out
        ]

        cursor = self.connection.cursor()
        created_tables = []
        failed_tables = []

        try:
            for i, query in enumerate(queries):
                table_name = table_names[i]
                try:
                    cursor.execute(query)
                    created_tables.append(table_name)
                    logger.info(f"✓ Created table: {table_name}")
                except Exception as e:
                    failed_tables.append((table_name, str(e)))
                    logger.error(f"✗ Failed to create table {table_name}: {e}")

            self.connection.commit()
            logger.info(f"Successfully created {len(created_tables)} tables")

            if failed_tables:
                logger.warning(f"Failed to create {len(failed_tables)} tables")
                for table_name, error in failed_tables:
                    logger.warning(f"  - {table_name}: {error}")

        except Exception as e:
            self.connection.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            cursor.close()

    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


def main():
    """Main function to create tables"""
    print("HR Management System - Database Table Creator")
    print("=" * 50)

    # Choose database type
    db_type = input("Choose database type (postgresql/sqlite) [postgresql]: ").strip().lower()
    if not db_type:
        db_type = "postgresql"

    creator = DatabaseTableCreator(db_type)

    try:
        if db_type == "postgresql":
            print("\nPostgreSQL Connection Details:")
            host = input("Host [localhost]: ").strip() or "localhost"
            database = input("Database name: ").strip()
            user = input("Username: ").strip()
            password = input("Password: ").strip()
            port = input("Port [5433]: ").strip() or "5433"

            if not database or not user:
                print("Database name and username are required!")
                return

            creator.connect_postgresql(host, database, user, password, int(port))

        else:  # SQLite
            db_path = input("Database file path [hr_management.db]: ").strip()
            if not db_path:
                db_path = "hr_management.db"
            creator.connect_sqlite(db_path)

        print("\nCreating tables...")
        creator.create_tables()
        print("\n✓ Table creation process completed!")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n✗ Error: {e}")
    finally:
        creator.close_connection()


if __name__ == "__main__":
    main()