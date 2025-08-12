import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
from datetime import datetime
import os
import re  # Add this import for regex operations
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseSeeder2:
    def __init__(self, db_config: Dict[str, str]):
        """Initialize database connection"""
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")

    def execute_query(self, query: str, params=None):
        """Execute a query with optional parameters"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Query execution failed: {e}")
            raise

    def check_table_constraints(self, table_name: str):
        """Check what constraints exist on a table"""
        query = """
        SELECT 
            conname as constraint_name,
            contype as constraint_type,
            array_agg(attname ORDER BY attnum) as columns
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_attribute att ON att.attrelid = con.conrelid 
            AND att.attnum = ANY(con.conkey)
        WHERE rel.relname = %s
        GROUP BY conname, contype
        """
        try:
            self.cursor.execute(query, (table_name,))
            return self.cursor.fetchall()
        except Exception as e:
            logger.warning(f"Could not check constraints for {table_name}: {e}")
            return []

    def bulk_insert_safe(self, table: str, columns: List[str], data: List[tuple],
                        primary_key_columns: Optional[List[str]] = None):
        """Perform bulk insert with safe conflict handling"""
        if not data:
            logger.warning(f"No data to insert into {table}")
            return

        # Log the columns and first row of data for debugging
        logger.info(f"Attempting to insert into {table} with columns: {columns}")
        if data:
            logger.info(f"First row data: {data[0]}")

        # Check existing constraints
        constraints = self.check_table_constraints(table)
        logger.info(f"Found constraints for {table}: {constraints}")

        # Try to find a suitable unique constraint
        unique_constraints = [c for c in constraints if c[1] == 'u']  # 'u' for unique
        primary_constraints = [c for c in constraints if c[1] == 'p']  # 'p' for primary key

        # Base insert query
        base_query = f"INSERT INTO {table} ({','.join(columns)}) VALUES %s"

        # Handle different tables differently
        if table == 'employee_exit':
            # For employee_exit, we want to update if the record exists
            conflict_cols = ['employee_code']
            update_cols = ['exit_date', 'last_working_date', 'exit_reason', 'exit_comments']
            conflict_action = f"ON CONFLICT (employee_code) DO UPDATE SET " + \
                            ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)
            query = f"{base_query} {conflict_action}"
        elif table == 'attendance':
            # For attendance, update if record exists for same employee and date
            conflict_cols = ['employee_code', 'attendance_date']
            update_cols = ['clock_in_time', 'clock_out_time', 'attendance_type']
            conflict_action = f"ON CONFLICT (employee_code, attendance_date) DO UPDATE SET " + \
                            ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)
            query = f"{base_query} {conflict_action}"
        elif table == 'timesheet':
            # For timesheet, update if record exists for same employee, project and date
            conflict_cols = ['employee_code', 'project_id', 'work_date']
            update_cols = ['hours_worked', 'task_description']
            conflict_action = f"ON CONFLICT (employee_code, project_id, work_date) DO UPDATE SET " + \
                            ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)
            query = f"{base_query} {conflict_action}"
        elif primary_constraints:
            # Use primary key constraint
            pk_columns = primary_constraints[0][2]  # Get columns from first primary key
            conflict_clause = f"ON CONFLICT ({','.join(pk_columns)}) DO NOTHING"
            query = f"{base_query} {conflict_clause}"
        elif unique_constraints:
            # Use first unique constraint found
            unique_columns = unique_constraints[0][2]
            conflict_clause = f"ON CONFLICT ({','.join(unique_columns)}) DO NOTHING"
            query = f"{base_query} {conflict_clause}"
        elif primary_key_columns:
            # Use specified primary key columns
            conflict_clause = f"ON CONFLICT ({','.join(primary_key_columns)}) DO NOTHING"
            query = f"{base_query} {conflict_clause}"
        else:
            # No constraints found, use simple insert
            query = base_query

        try:
            execute_values(self.cursor, query, data, template=None, page_size=1000)
            self.conn.commit()
            logger.info(f"Successfully processed {len(data)} records for {table}")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Bulk insert failed for {table}: {e}")
            # Try individual inserts to identify problematic records
            self._try_individual_inserts(table, columns, data)

    def _try_individual_inserts(self, table: str, columns: List[str], data: List[tuple]):
        """Try individual inserts to handle duplicates gracefully"""
        logger.info(f"Attempting individual inserts for {table}")
        success_count = 0
        failure_count = 0

        insert_query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))})"

        for record in data:
            try:
                self.cursor.execute(insert_query, record)
                self.conn.commit()
                success_count += 1
            except Exception as e:
                self.conn.rollback()
                failure_count += 1
                if failure_count <= 5:
                    logger.warning(f"Failed to insert record {record[:2]}...: {e}")

        logger.info(f"Individual inserts completed: {success_count} success, {failure_count} failures")

    def seed_departments_and_designations(self, df_emp: pd.DataFrame):
        """Seed departments and designations from employee data"""
        logger.info("Seeding departments and designations...")

        # Extract unique departments
        departments = df_emp[['Department', 'Business Unit', 'Parent Department']].drop_duplicates()
        dept_data = []
        for _, row in departments.iterrows():
            dept_data.append((
                row['Department'],
                row['Business Unit'],
                row['Parent Department'] if pd.notna(row['Parent Department']) else None,
                'Active'
            ))

        # Use the safe bulk insert method
        self.bulk_insert_safe('department',
                            ['department_name', 'business_unit', 'parent_department', 'status'],
                            dept_data)

        # Extract unique designations
        designations = df_emp['Designation'].unique()
        desig_data = [(desig, 'Mid', 'Active') for desig in designations if pd.notna(desig)]

        self.bulk_insert_safe('designation',
                            ['designation_name', 'level', 'status'],
                            desig_data)

    def get_reference_mappings(self):
        """Get department and designation ID mappings"""
        # Get department mappings
        self.cursor.execute("SELECT department_id, department_name FROM department")
        dept_mapping = {name: dept_id for dept_id, name in self.cursor.fetchall()}

        # Get designation mappings
        self.cursor.execute("SELECT designation_id, designation_name FROM designation")
        desig_mapping = {name: desig_id for desig_id, name in self.cursor.fetchall()}

        return dept_mapping, desig_mapping

    def parse_date(self, date_str: str) -> datetime.date:
        """Parse date string in multiple formats"""
        if pd.isna(date_str):
            return None
            
        formats = ['%d-%m-%Y', '%Y-%m-%d']  # Add more formats if needed
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt).date()
            except ValueError:
                continue
                
        # If no format works, try pandas' flexible parser
        try:
            return pd.to_datetime(date_str).date()
        except Exception as e:
            logger.error(f"Failed to parse date {date_str}: {e}")
            return None

    def parse_experience_value(self, exp_str: str) -> float:
        """Parse experience string like '4 years 1 months 25 days' to decimal years"""
        if pd.isna(exp_str) or exp_str == '':
            return 0.0
        
        try:
            # Handle string format like "4 years 1 months 25 days"
            exp_str = str(exp_str).lower()
            years = 0
            months = 0
            days = 0
            
            # Extract years
            if 'year' in exp_str:
                years_match = re.search(r'(\d+)\s*year', exp_str)
                if years_match:
                    years = int(years_match.group(1))
            
            # Extract months
            if 'month' in exp_str:
                months_match = re.search(r'(\d+)\s*month', exp_str)
                if months_match:
                    months = int(months_match.group(1))
            
            # Extract days
            if 'day' in exp_str:
                days_match = re.search(r'(\d+)\s*day', exp_str)
                if days_match:
                    days = int(days_match.group(1))
            
            # Convert to decimal years
            total_years = years + (months / 12.0) + (days / 365.0)
            return round(total_years, 2)
            
        except Exception as e:
            logger.warning(f"Failed to parse experience '{exp_str}': {e}")
            return 0.0

    def get_safe_value(self, row, column_name: str, default_value=None):
        """Safely get value from DataFrame row, handling missing columns"""
        try:
            if column_name in row.index and pd.notna(row[column_name]):
                return row[column_name]
            return default_value
        except Exception:
            return default_value

    def seed_employees(self, df_emp: pd.DataFrame, dept_mapping: Dict, desig_mapping: Dict):
        """Seed employee data with updated column handling"""
        logger.info("Seeding employees...")

        emp_data = []
        personal_data = []
        financial_data = []

        # Keep track of used Aadhaar and PAN numbers
        used_aadhaar = {}
        used_pan = {}

        for _, row in df_emp.iterrows():
            # Handle status - map "Inactive" correctly
            status = self.get_safe_value(row, 'Status', 'Active')
            if status and status.strip().lower() == 'inactive':
                status = 'Inactive'
            else:
                status = 'Active'

            # Main employee data with safe value extraction
            emp_data.append((
                self.get_safe_value(row, 'Employee Code'),
                self.get_safe_value(row, 'Employee Name'),
                self.get_safe_value(row, 'Email'),
                self.get_safe_value(row, 'Mobile Number'),
                self.parse_date(self.get_safe_value(row, 'Date Of Joining')),
                self.get_safe_value(row, 'Employee Type', 'Regular'),
                self.get_safe_value(row, 'Grade'),
                status,
                dept_mapping.get(self.get_safe_value(row, 'Department')),
                self.get_safe_value(row, 'Department'),
                desig_mapping.get(self.get_safe_value(row, 'Designation')),
                None,  # primary_manager_id
                0.0,   # past_experience - will be updated later
                0.0    # current_experience - will be updated later
            ))

            # Handle duplicate Aadhaar numbers
            aadhaar = self.get_safe_value(row, 'Aadhaar Number')
            if aadhaar:
                aadhaar = str(aadhaar)
                if aadhaar in used_aadhaar:
                    base = aadhaar[:-1] if len(aadhaar) > 1 else aadhaar
                    last_digit = int(aadhaar[-1]) if aadhaar[-1].isdigit() else 0
                    new_last_digit = (last_digit + 1) % 10
                    aadhaar = f"{base}{new_last_digit}"
                    while aadhaar in used_aadhaar.values():
                        new_last_digit = (new_last_digit + 1) % 10
                        aadhaar = f"{base}{new_last_digit}"
                used_aadhaar[str(self.get_safe_value(row, 'Aadhaar Number', ''))] = aadhaar

            # Handle duplicate PAN numbers
            pan = self.get_safe_value(row, 'PAN Number')
            if pan:
                pan = str(pan)
                if pan in used_pan:
                    base = pan[:-1] if len(pan) > 1 else pan
                    last_char = pan[-1] if len(pan) > 0 else 'A'
                    if last_char.isdigit():
                        new_last_char = str((int(last_char) + 1) % 10)
                    else:
                        new_last_char = chr((ord(last_char.upper()) - ord('A') + 1) % 26 + ord('A'))
                    pan = f"{base}{new_last_char}"
                    while pan in used_pan.values():
                        if new_last_char.isdigit():
                            new_last_char = str((int(new_last_char) + 1) % 10)
                        else:
                            new_last_char = chr((ord(new_last_char) - ord('A') + 1) % 26 + ord('A'))
                        pan = f"{base}{new_last_char}"
                used_pan[str(self.get_safe_value(row, 'PAN Number', ''))] = pan

            # Personal data with safe extraction
            personal_data.append((
                self.get_safe_value(row, 'Employee Code'),
                self.get_safe_value(row, 'Gender'),
                self.parse_date(self.get_safe_value(row, 'Date Of Birth')),
                self.get_safe_value(row, 'Marital Status'),
                self.get_safe_value(row, 'Present Address'),
                self.get_safe_value(row, 'Permanent Address'),
                pan,
                aadhaar
            ))

            # Financial data with safe extraction
            financial_data.append((
                self.get_safe_value(row, 'Employee Code'),
                self.get_safe_value(row, 'Bank Name'),
                self.get_safe_value(row, 'Account Number'),
                self.get_safe_value(row, 'IFSC Code')
            ))

        # Insert using existing bulk_insert_safe method
        self.bulk_insert_safe('employee',
                            ['employee_code', 'employee_name', 'email', 'mobile_number',
                             'date_of_joining', 'employee_type', 'grade', 'status',
                             'department_id', 'department_name', 'designation_id', 'primary_manager_id',
                             'past_experience', 'current_experience'],
                            emp_data,
                            primary_key_columns=['employee_code'])

        self.bulk_insert_safe('employee_personal',
                            ['employee_code', 'gender', 'date_of_birth', 'marital_status',
                             'present_address', 'permanent_address', 'pan_number', 'aadhaar_number'],
                            personal_data,
                            primary_key_columns=['employee_code'])

        self.bulk_insert_safe('employee_financial',
                            ['employee_code', 'bank_name', 'account_number', 'ifsc_code'],
                            financial_data,
                            primary_key_columns=['employee_code'])

    def update_experience_data(self, df_exp: pd.DataFrame):
        """Update employee experience data with proper parsing"""
        logger.info("Updating employee experience data...")

        for _, row in df_exp.iterrows():
            current_exp = self.parse_experience_value(self.get_safe_value(row, 'Current Experience', '0'))
            past_exp = self.parse_experience_value(self.get_safe_value(row, 'Past Experience', '0'))
            
            update_query = """
                UPDATE employee 
                SET current_experience = %s,
                    past_experience = %s
                WHERE employee_code = %s
            """
            try:
                self.execute_query(update_query, (
                    current_exp,
                    past_exp,
                    self.get_safe_value(row, 'Employee Code')
                ))
            except Exception as e:
                logger.warning(f"Failed to update experience for {self.get_safe_value(row, 'Employee Code')}: {e}")

    def seed_work_profiles(self, df_work: pd.DataFrame):
        """Seed employee work profile data"""
        logger.info("Seeding work profiles...")

        work_profile_data = []
        for _, row in df_work.iterrows():
            # Extract and process the data
            designation = self.get_safe_value(row, 'Designation', 'Not Specified')
            department = self.get_safe_value(row, 'Department', 'Not Specified')
            business_unit = self.get_safe_value(row, 'Business Unit', 'Not Specified')

            # Combine business unit, department and designation for skills categorization
            primary_skills = f"{business_unit},{department}"
            secondary_skills = designation

            # For new employees, start with 0 years experience
            # These will be updated through the experience update process
            work_profile_data.append((
                self.get_safe_value(row, 'Employee Code'),
                designation,  # Use designation as role
                0.0,  # Total experience - will be calculated
                0.0,  # Relevant experience - will be calculated
                '',  # Certifications - to be added later
                '',  # Past projects - to be added later
                primary_skills,
                secondary_skills,
                'Active'
            ))

            logger.info(f"Processing work profile for employee: {row['Employee Code']}")

        self.bulk_insert_safe('employee_work_profile',
                            ['employee_code', 'role', 'total_experience_years',
                             'relevant_experience_years', 'certifications', 'past_projects',
                             'primary_skills', 'secondary_skills', 'status'],
                            work_profile_data,
                            primary_key_columns=['employee_code'])
        
        # Update employee table with initial experience as 0
        # This can be updated later through the experience update process
        for _, row in df_work.iterrows():
            try:
                update_query = """
                    UPDATE employee 
                    SET past_experience = %s,
                        current_experience = %s,
                        department_name = %s
                    WHERE employee_code = %s
                """
                self.execute_query(update_query, (
                    0.0,  # past_experience
                    0.0,  # current_experience
                    self.get_safe_value(row, 'Department', 'Not Specified'),
                    row['Employee Code']
                ))
            except Exception as e:
                logger.warning(f"Failed to update work profile for {row['Employee Code']}: {e}")

    def seed_employee_exits(self, df_exit: pd.DataFrame):
        """Seed employee exit data"""
        logger.info("Seeding employee exits...")

        exit_data = []
        for _, row in df_exit.iterrows():
            exit_data.append((
                row['Employee Code'],
                pd.to_datetime(row['Exit Date']).date(),
                pd.to_datetime(row['Expected Resignation Date']).date(),
                'Resignation',  # default reason
                f"Employee {row['Employee Name']} resigned"
            ))

        self.bulk_insert_safe('employee_exit',
                            ['employee_code', 'exit_date', 'last_working_date',
                             'exit_reason', 'exit_comments'],
                            exit_data,
                            primary_key_columns=['employee_code'])

        # Update employee status to Inactive for exited employees
        for _, row in df_exit.iterrows():
            try:
                update_query = "UPDATE employee SET status = 'Inactive' WHERE employee_code = %s"
                self.execute_query(update_query, (row['Employee Code'],))
            except Exception as e:
                logger.warning(f"Failed to update status for exited employee {row['Employee Code']}: {e}")

    def seed_projects(self, df_timesheet: pd.DataFrame):
        """Seed project data from timesheet data"""
        logger.info("Seeding projects...")

        # First check existing projects
        project_query = """
            SELECT project_id, project_name
            FROM project
            WHERE status = 'Active'
        """
        self.cursor.execute(project_query)
        existing_projects = {pid: name for pid, name in self.cursor.fetchall()}

        # Process new projects from timesheet
        project_data = []
        for project_id in df_timesheet['project_id'].unique():
            if project_id not in existing_projects:
                project_data.append((
                    project_id,
                    f"Project {project_id}",  # Default project name since it's not in the CSV
                    'Default Client',  # placeholder client name
                    'Active',
                    datetime.now().date(),  # start_date
                    None  # end_date
                ))

        if project_data:
            self.bulk_insert_safe('project',
                                ['project_id', 'project_name', 'client_name', 'status',
                                 'start_date', 'end_date'],
                                project_data,
                                primary_key_columns=['project_id'])

    def seed_attendance(self, df_attendance: pd.DataFrame):
        """Seed attendance data with updated column names"""
        logger.info("Seeding attendance data...")

        attendance_data = []
        for _, row in df_attendance.iterrows():
            try:
                # Use ShiftDate instead of Date
                shift_date = self.get_safe_value(row, 'ShiftDate')
                in_time = self.get_safe_value(row, 'In Time')
                out_time = self.get_safe_value(row, 'Out Time')
                status = self.get_safe_value(row, 'Status', 'Present')

                if shift_date:
                    attendance_data.append((
                        pd.to_datetime(shift_date).date(),
                        self.get_safe_value(row, 'Employee Code'),
                        pd.to_datetime(in_time, format='%H:%M:%S').time() if in_time else None,
                        pd.to_datetime(out_time, format='%H:%M:%S').time() if out_time else None,
                        status
                    ))
            except Exception as e:
                logger.warning(f"Failed to parse attendance record: {e}")
                continue

        self.bulk_insert_safe('attendance',
                            ['attendance_date', 'employee_code', 'clock_in_time',
                             'clock_out_time', 'attendance_type'],
                            attendance_data,
                            primary_key_columns=['employee_code', 'attendance_date'])

    def seed_timesheets(self, df_timesheet: pd.DataFrame):
        """Seed timesheet data"""
        logger.info("Seeding timesheet data...")

        timesheet_data = []
        for _, row in df_timesheet.iterrows():
            try:
                timesheet_data.append((
                    pd.to_datetime(row['work_date']).date(),
                    row['employee_code'],
                    row['project_id'],
                    float(row['hours_worked']),
                    row['task_description'] if pd.notna(row['task_description']) else None
                ))
            except Exception as e:
                logger.warning(f"Failed to parse timesheet record: {e}")
                continue

        self.bulk_insert_safe('timesheet',
                            ['work_date', 'employee_code', 'project_id', 'hours_worked',
                             'task_description'],
                            timesheet_data,
                            primary_key_columns=['employee_code', 'project_id', 'work_date'])

    def seed_project_allocations(self, df_allocations: pd.DataFrame, csv_files: Dict[str, str] = None):
        """Seed project allocation data"""
        logger.info("Seeding project allocations...")

        # First ensure all projects exist
        project_data = []
        seen_projects = set()
        for _, row in df_allocations.iterrows():
            if row['Project Code'] not in seen_projects:
                project_data.append((
                    row['Project Code'],
                    row['Project Name'],
                    None,  # client_name
                    'Active',  # status
                    pd.to_datetime(row['Available From']).date(),  # start_date
                    None,  # end_date
                ))
                seen_projects.add(row['Project Code'])
        
        # Insert projects
        if project_data:
            logger.info(f"Inserting {len(project_data)} projects")
            self.bulk_insert_safe('project',
                                ['project_id', 'project_name', 'client_name', 'status', 'start_date', 'end_date'],
                                project_data,
                                primary_key_columns=['project_id'])

        # Get employee codes from employee master
        try:
            if csv_files and 'employee_master' in csv_files:
                emp_df = pd.read_csv(csv_files['employee_master'])
            else:
                emp_df = pd.read_csv('updated_csv_files/employee_master.csv')  # fallback
            employee_map = dict(zip(emp_df['Employee Name'], emp_df['Employee Code']))
            logger.info(f"Loaded {len(employee_map)} employee mappings")
        except Exception as e:
            logger.error(f"Failed to read employee master data: {e}")
            raise

        # Now insert allocations
        allocation_data = []
        for _, row in df_allocations.iterrows():
            try:
                emp_code = employee_map.get(row['Name'])
                if not emp_code:
                    logger.warning(f"Could not find employee code for {row['Name']}")
                    continue

                # Use the mapped employee code
                allocation_data.append((
                    emp_code,  # employee_code
                    row['Project Code'],  # project_id
                    float(row['% Allocation']) if pd.notna(row['% Allocation']) else 100.0,
                    pd.to_datetime(row['Available From']).date(),
                    None,  # effective_to
                    'Active',  # status
                    'system',  # created_by
                    row['Comments'] if pd.notna(row['Comments']) else None
                ))
            except Exception as e:
                logger.warning(f"Failed to parse project allocation record for employee={row.get('Name')} project_id={row.get('Project Code')}: {e}")
                continue

        if allocation_data:
            try:
                logger.info(f"Attempting to insert {len(allocation_data)} project allocations")
                # Use the correct column names here!
                self.bulk_insert_safe('project_allocation',
                                    ['employee_code', 'project_id', 'allocation_percentage',
                                     'effective_from', 'effective_to', 'status', 'created_by',
                                     'change_reason'],
                                    allocation_data)
            except Exception as e:
                logger.error(f"Bulk insert failed for project allocations: {e}")
                logger.info("Attempting individual inserts...")
                success_count = 0
                for record in allocation_data:
                    try:
                        self.cursor.execute(
                            """
                            INSERT INTO project_allocation 
                            (employee_code, project_id, allocation_percentage, effective_from, 
                             effective_to, status, created_by, change_reason) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            record
                        )
                        self.conn.commit()
                        success_count += 1
                    except Exception as ex:
                        logger.error(f"Failed to insert allocation for employee_code={record[0]}, project_id={record[1]}: {ex}")
                logger.info(f"Individual inserts completed: {success_count} successful out of {len(allocation_data)}")

    def seed_database(self, csv_files: Dict[str, str], clean_existing: bool = False,
                 tables_to_clean: List[str] = None):
        """Seed database with data from CSV files"""
        try:
            logger.info(f"Starting database seeding with files: {list(csv_files.keys())}")
            
            if clean_existing:
                self.clean_existing_data(tables_to_clean)

            # Load employee master first if available
            df_emp = None
            if 'employee_master' in csv_files and csv_files['employee_master'] is not None:
                df_emp = csv_files['employee_master']
                if isinstance(df_emp, str):  # If it's a file path
                    logger.info(f"Reading employee_master from file: {df_emp}")
                    df_emp = pd.read_csv(df_emp)
                logger.info(f"Processing {len(df_emp)} employee records")
                
                # First seed departments and designations
                self.seed_departments_and_designations(df_emp)
                
                # Get reference mappings
                dept_mapping, desig_mapping = self.get_reference_mappings()
                logger.info(f"Found {len(dept_mapping)} departments and {len(desig_mapping)} designations")
                
                # Then seed employees
                self.seed_employees(df_emp, dept_mapping, desig_mapping)

            # Update work profiles
            if 'work_profile' in csv_files and csv_files['work_profile'] is not None:
                df_work = csv_files['work_profile']
                if isinstance(df_work, str):
                    logger.info(f"Reading work_profile from file: {df_work}")
                    df_work = pd.read_csv(df_work)
                logger.info(f"Processing {len(df_work)} work profiles")
                self.seed_work_profiles(df_work)

            # Update experience data
            if 'experience_report' in csv_files and csv_files['experience_report'] is not None:
                df_exp = csv_files['experience_report']
                if isinstance(df_exp, str):
                    logger.info(f"Reading experience_report from file: {df_exp}")
                    df_exp = pd.read_csv(df_exp)
                logger.info(f"Processing {len(df_exp)} experience records")
                self.update_experience_data(df_exp)

            # Seed employee exits
            if 'employee_exit' in csv_files and csv_files['employee_exit'] is not None:
                df_exit = csv_files['employee_exit']
                if isinstance(df_exit, str):
                    logger.info(f"Reading employee_exit from file: {df_exit}")
                    df_exit = pd.read_csv(df_exit)
                logger.info(f"Processing {len(df_exit)} exit records")
                self.seed_employee_exits(df_exit)

            # Seed projects from timesheet data
            if 'timesheet_report' in csv_files and csv_files['timesheet_report'] is not None:
                df_timesheet = csv_files['timesheet_report']
                if isinstance(df_timesheet, str):
                    logger.info(f"Reading timesheet_report from file: {df_timesheet}")
                    df_timesheet = pd.read_csv(df_timesheet)
                logger.info(f"Processing {len(df_timesheet)} timesheet records")
                self.seed_projects(df_timesheet)

            # Seed project allocations
            if 'project_allocations' in csv_files and csv_files['project_allocations'] is not None:
                df_allocations = csv_files['project_allocations']
                if isinstance(df_allocations, str):
                    logger.info(f"Reading project_allocations from file: {df_allocations}")
                    df_allocations = pd.read_csv(df_allocations)
                logger.info(f"Processing {len(df_allocations)} allocation records")
                self.seed_project_allocations(df_allocations, csv_files)

            # Seed resource utilization
            if 'resource_utilization' in csv_files and csv_files['resource_utilization'] is not None:
                df_utilization = csv_files['resource_utilization']
                if isinstance(df_utilization, str):
                    logger.info(f"Reading resource_utilization from file: {df_utilization}")
                    df_utilization = pd.read_csv(df_utilization)
                logger.info(f"Processing {len(df_utilization)} resource utilization records")
                self.seed_resource_utilization(df_utilization)

            # Seed attendance data
            if 'attendance_report' in csv_files and csv_files['attendance_report'] is not None:
                df_attendance = csv_files['attendance_report']
                if isinstance(df_attendance, str):
                    logger.info(f"Reading attendance_report from file: {df_attendance}")
                    df_attendance = pd.read_csv(df_attendance)
                logger.info(f"Processing {len(df_attendance)} attendance records")
                self.seed_attendance(df_attendance)

            # Finally seed timesheet data
            if 'timesheet_report' in csv_files and csv_files['timesheet_report'] is not None:
                logger.info(f"Processing {len(df_timesheet)} timesheet entries")
                self.seed_timesheets(df_timesheet)
                
            if 'resource_utilization' in csv_files and csv_files['resource_utilization'] is not None:
                logger.info(f"Processing {len(df_utilization)} resource utilization entries")
                self.seed_resource_utilization(df_utilization)

            logger.info("Database seeding completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error seeding database: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    def seed_resource_utilization(self, df_utilization: pd.DataFrame):
        """Seed resource utilization data"""
        logger.info("Seeding resource utilization data...")

        utilization_data = []
        for _, row in df_utilization.iterrows():
            try:
                utilization_data.append((
                    row['project_id'],
                    pd.to_datetime(row['week_start_date']).date(),
                    float(row['estimated_hours']) if pd.notna(row['estimated_hours']) else 0.0
                ))
            except Exception as e:
                logger.warning(f"Failed to parse resource utilization record: {e}")
                continue

        self.bulk_insert_safe('resource_utilization',
                            ['project_id', 'week_start_date', 'estimated_hours'],
                            utilization_data,
                            primary_key_columns=['project_id', 'week_start_date'])

def main():
    # Database configuration from docker-compose
    db_config = {
        'host': 'localhost',
        'port': '5434',
        'database': 'employee_db',
        'user': 'postgres',
        'password': 'postgres123'
    }

    # CSV files mapping with actual filenames
    csv_files = {
        'employee_master': 'updated_csv_files/employee_master.csv',
        'employee_exit': 'updated_csv_files/exit-report.csv',
        'experience_report': 'updated_csv_files/experience_report.csv',
        'work_profile': 'updated_csv_files/employee_work_profile.csv',
        'attendance_report': 'updated_csv_files/attendance-report.csv',
        'timesheet_report': 'updated_csv_files/timesheets.csv',
        'project_allocations': 'updated_csv_files/allocations.csv'
    }

    # Verify CSV files exist
    for file_type, file_path in csv_files.items():
        if not os.path.exists(file_path):
            logger.warning(f"CSV file {file_path} not found. Skipping {file_type} data.")
            csv_files[file_type] = None

    # Remove None values
    csv_files = {k: v for k, v in csv_files.items() if v is not None}

    if not csv_files:
        logger.error("No CSV files found. Please ensure CSV files are in the current directory.")
        return

    # Initialize and run seeder
    seeder = DatabaseSeeder2(db_config)
    try:
        seeder.connect()
        seeder.seed_database(csv_files)
    finally:
        seeder.disconnect()


if __name__ == "__main__":
    main()