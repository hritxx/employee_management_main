#!/usr/bin/env python3
"""
Project and Allocation Data Seeder
Seeds the database with project and allocation data from CSV files.

Author: AI Data Scientist with 30+ years of experience
"""

import pandas as pd
import logging
import re
from datetime import datetime, date
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import sys
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import get_cursor, get_connection
from config.config import db_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/data_seeding_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProjectDataSeeder:
    """
    Advanced data seeding class for projects and allocations.
    Implements enterprise-grade data validation, error handling, and transaction management.
    """
    
    def __init__(self, csv_file_path: str = "allocation/allocations.csv"):
        """
        Initialize the seeder with CSV file path and validation rules.
        
        Args:
            csv_file_path: Path to the allocations CSV file
        """
        self.csv_file_path = Path(csv_file_path)
        self.data_df: Optional[pd.DataFrame] = None
        self.validation_errors: List[Dict] = []
        self.project_type_mapping = {
            'Development': 'DEV',
            'Infrastructure': 'INFRA', 
            'Research': 'RESEARCH',
            'Security': 'SEC',
            'Analytics': 'ANALYTICS'
        }
        
        # Employee code mapping - will be dynamically loaded from database
        self.employee_mapping: Dict[str, str] = {}
        self.employee_name_conflicts: List[Dict] = []
        
        # Parsing statistics tracking
        self.parsing_stats = {
            'allocation_formats_found': set(),
            'date_formats_found': set(),
            'allocation_parsing_failures': [],
            'date_parsing_failures': [],
            'successful_allocation_parses': 0,
            'successful_date_parses': 0
        }
        
    def load_employee_mapping(self) -> bool:
        """
        Load employee name to code mapping from database.
        Handles name conflicts and validates data integrity.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with get_cursor() as cursor:
                # Get all active employees
                cursor.execute("""
                    SELECT employee_code, employee_name, email, status 
                    FROM employee 
                    WHERE status = 'Active'
                    ORDER BY employee_name
                """)
                
                employees = cursor.fetchall()
                logger.info(f"Found {len(employees)} active employees in database")
                
                # Build mapping and detect conflicts
                name_counts: Dict[str, List[Tuple[str, str]]] = {}
                
                for emp_code, emp_name, email, status in employees:
                    # Normalize name for better matching
                    normalized_name = self._normalize_employee_name(emp_name)
                    
                    if normalized_name not in name_counts:
                        name_counts[normalized_name] = []
                    name_counts[normalized_name].append((emp_code, emp_name))
                
                # Handle name conflicts and build final mapping
                for normalized_name, emp_list in name_counts.items():
                    if len(emp_list) == 1:
                        # No conflict - direct mapping
                        emp_code, original_name = emp_list[0]
                        self.employee_mapping[normalized_name] = emp_code
                        # Also map the original name as stored in DB
                        self.employee_mapping[original_name] = emp_code
                        logger.debug(f"Mapped: '{normalized_name}' → {emp_code}")
                    else:
                        # Conflict detected
                        self.employee_name_conflicts.append({
                            'normalized_name': normalized_name,
                            'employees': emp_list,
                            'count': len(emp_list)
                        })
                        logger.warning(f"Name conflict detected: '{normalized_name}' maps to {len(emp_list)} employees")
                        for emp_code, original_name in emp_list:
                            logger.warning(f"  → {emp_code}: {original_name}")
                
                # Log conflicts
                if self.employee_name_conflicts:
                    logger.error(f"Found {len(self.employee_name_conflicts)} name conflicts that need resolution")
                    return False
                
                logger.info(f"Successfully loaded {len(self.employee_mapping)} employee mappings")
                return True
                
        except Exception as e:
            logger.error(f"Error loading employee mapping: {e}")
            return False
    
    def _normalize_employee_name(self, name: str) -> str:
        """
        Normalize employee name for consistent matching.
        
        Args:
            name: Original employee name
            
        Returns:
            str: Normalized name
        """
        if not name:
            return ""
        
        # Convert to title case and strip whitespace
        normalized = name.strip().title()
        
        # Remove extra whitespace between words
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _parse_allocation_percentage(self, allocation_value) -> Optional[float]:
        """
        Parse allocation percentage from various formats.
        
        Handles formats like:
        - 0.8 (converts to 80)
        - 80% (extracts 80)
        - 80 (returns 80)
        - "80%" (extracts 80)
        - "0.8" (converts to 80)
        
        Args:
            allocation_value: Raw allocation value from CSV
            
        Returns:
            Optional[float]: Parsed allocation percentage or None if invalid
        """
        if pd.isna(allocation_value) or allocation_value == '':
            return None
        
        original_value = allocation_value
        
        try:
            # Convert to string and clean up
            value_str = str(allocation_value).strip()
            
            # Track the original format
            self.parsing_stats['allocation_formats_found'].add(str(original_value))
            
            # Handle percentage format (remove % sign)
            if '%' in value_str:
                value_str = value_str.replace('%', '').strip()
            
            # Convert to float
            parsed_value = float(value_str)
            
            # If value is between 0 and 1, assume it's a decimal representation (e.g., 0.8 = 80%)
            if 0 <= parsed_value <= 1:
                parsed_value = parsed_value * 100
            
            # Validate range
            if 0 <= parsed_value <= 100:
                self.parsing_stats['successful_allocation_parses'] += 1
                return parsed_value
            else:
                logger.warning(f"Allocation percentage {parsed_value} is outside valid range (0-100)")
                self.parsing_stats['allocation_parsing_failures'].append(f"{original_value} -> {parsed_value} (out of range)")
                return None
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse allocation percentage '{allocation_value}': {e}")
            self.parsing_stats['allocation_parsing_failures'].append(f"{original_value} -> {str(e)}")
            return None
    
    def _parse_date_flexible(self, date_value) -> Optional[pd.Timestamp]:
        """
        Parse date from various formats with flexible handling.
        
        Handles formats like:
        - "3rd December" or "December 3rd"
        - "December 3" or "3 December" 
        - "2024-12-03"
        - "12/03/2024"
        - "03-Dec-2024"
        
        Args:
            date_value: Raw date value from CSV
            
        Returns:
            Optional[pd.Timestamp]: Parsed date or None if invalid
        """
        if pd.isna(date_value) or date_value == '':
            return None
        
        original_value = date_value
        
        try:
            # Convert to string and clean up
            date_str = str(date_value).strip()
            
            # Track the original format
            self.parsing_stats['date_formats_found'].add(str(original_value))
            
            # Handle ordinal numbers (1st, 2nd, 3rd, etc.)
            ordinal_pattern = r'\b(\d+)(st|nd|rd|th)\b'
            date_str = re.sub(ordinal_pattern, r'\1', date_str, flags=re.IGNORECASE)
            
            # Try various date parsing strategies
            parsing_strategies = [
                # Strategy 1: Use pandas default parsing (handles most standard formats)
                lambda x: pd.to_datetime(x, infer_datetime_format=True),
                
                # Strategy 2: Try with dayfirst=True for DD/MM/YYYY formats
                lambda x: pd.to_datetime(x, dayfirst=True),
                
                # Strategy 3: Try specific formats for text dates
                lambda x: pd.to_datetime(x, format='%B %d'),  # "December 3"
                lambda x: pd.to_datetime(x, format='%d %B'),  # "3 December"
                lambda x: pd.to_datetime(x, format='%B %d, %Y'),  # "December 3, 2024"
                lambda x: pd.to_datetime(x, format='%d %B %Y'),  # "3 December 2024"
                
                # Strategy 4: Try with fuzzy parsing (requires dateutil)
                lambda x: pd.to_datetime(x, errors='coerce')
            ]
            
            for strategy in parsing_strategies:
                try:
                    result = strategy(date_str)
                    if pd.notna(result):
                        # If year is missing, assume current year
                        if result.year == 1900:  # pandas default for missing year
                            current_year = datetime.now().year
                            result = result.replace(year=current_year)
                        
                        self.parsing_stats['successful_date_parses'] += 1
                        return result
                except:
                    continue
            
            # If all strategies fail, log the issue
            logger.warning(f"Could not parse date '{date_value}' using any strategy")
            self.parsing_stats['date_parsing_failures'].append(str(original_value))
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing date '{date_value}': {e}")
            self.parsing_stats['date_parsing_failures'].append(f"{original_value} -> {str(e)}")
            return None
    
    def get_employee_code_for_name(self, csv_name: str) -> Optional[str]:
        """
        Get employee code for a name from CSV, with fuzzy matching support.
        
        Args:
            csv_name: Employee name from CSV
            
        Returns:
            Optional[str]: Employee code if found, None otherwise
        """
        if not csv_name:
            return None
        
        # Try exact match first
        normalized_csv_name = self._normalize_employee_name(csv_name)
        
        # Direct mapping lookup
        if normalized_csv_name in self.employee_mapping:
            return self.employee_mapping[normalized_csv_name]
        
        # Try original name as well
        if csv_name in self.employee_mapping:
            return self.employee_mapping[csv_name]
        
        # Try fuzzy matching for common variations
        return self._fuzzy_match_employee_name(normalized_csv_name)
    
    def _fuzzy_match_employee_name(self, csv_name: str) -> Optional[str]:
        """
        Attempt fuzzy matching for employee names to handle variations.
        
        Args:
            csv_name: Normalized CSV name
            
        Returns:
            Optional[str]: Employee code if fuzzy match found, None otherwise
        """
        # Try removing middle initials/names
        name_parts = csv_name.split()
        if len(name_parts) >= 3:
            # Try "First Last" combination
            simplified_name = f"{name_parts[0]} {name_parts[-1]}"
            if simplified_name in self.employee_mapping:
                logger.info(f"Fuzzy match: '{csv_name}' → '{simplified_name}' → {self.employee_mapping[simplified_name]}")
                return self.employee_mapping[simplified_name]
        
        # Try partial matches (be careful with this)
        for db_name, emp_code in self.employee_mapping.items():
            if len(name_parts) >= 2 and len(db_name.split()) >= 2:
                db_parts = db_name.split()
                # Check if first and last names match
                if (name_parts[0].lower() == db_parts[0].lower() and 
                    name_parts[-1].lower() == db_parts[-1].lower()):
                    logger.info(f"Partial match: '{csv_name}' → '{db_name}' → {emp_code}")
                    return emp_code
        
        return None

    def validate_csv_structure(self) -> bool:
        """
        Validate CSV file structure and required columns.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        required_columns = [
            'Project Type', 'Project Code', 'Project Name', 'Name', 
            '% Allocation', 'Role', 'Available From'
        ]
        
        try:
            if not self.csv_file_path.exists():
                logger.error(f"CSV file not found: {self.csv_file_path}")
                return False
                
            self.data_df = pd.read_csv(self.csv_file_path)
            logger.info(f"Loaded CSV with {len(self.data_df)} rows")
            
            # Check required columns
            missing_columns = set(required_columns) - set(self.data_df.columns)
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Validate data types and ranges
            self._validate_data_quality()
            
            return len(self.validation_errors) == 0
            
        except Exception as e:
            logger.error(f"Error validating CSV structure: {e}")
            return False
    
    def _validate_data_quality(self) -> None:
        """Validate data quality and log any issues using robust parsing."""
        for idx, row in self.data_df.iterrows():
            # Validate allocation percentage using robust parsing
            allocation = self._parse_allocation_percentage(row['% Allocation'])
            if allocation is None:
                self.validation_errors.append({
                    'row': idx + 2,  # +2 for header and 0-indexing
                    'field': '% Allocation',
                    'value': row['% Allocation'],
                    'error': 'Could not parse allocation percentage. Expected formats: 80, 80%, 0.8, "80%"'
                })
            elif not (0 <= allocation <= 100):
                self.validation_errors.append({
                    'row': idx + 2,
                    'field': '% Allocation',
                    'value': row['% Allocation'],
                    'error': f'Allocation percentage {allocation}% is outside valid range (0-100)'
                })
            
            # Validate employee mapping
            employee_code = self.get_employee_code_for_name(row['Name'])
            if not employee_code:
                self.validation_errors.append({
                    'row': idx + 2,
                    'field': 'Name',
                    'value': row['Name'],
                    'error': f'Employee name not found or ambiguous. Available employees: {len(self.employee_mapping)} loaded from database'
                })
            
            # Validate date format using robust parsing
            parsed_date = self._parse_date_flexible(row['Available From'])
            if parsed_date is None:
                self.validation_errors.append({
                    'row': idx + 2,
                    'field': 'Available From',
                    'value': row['Available From'],
                    'error': 'Could not parse date. Expected formats: "December 3", "3rd December", "2024-12-03", etc.'
                })
        
        if self.validation_errors:
            logger.warning(f"Found {len(self.validation_errors)} validation errors")
            for error in self.validation_errors[:5]:  # Show first 5 errors
                logger.warning(f"Row {error['row']}: {error['field']} = {error['value']} - {error['error']}")
    
    def extract_unique_projects(self) -> List[Dict]:
        """
        Extract unique projects from the CSV data.
        
        Returns:
            List of project dictionaries with standardized data
        """
        if self.data_df is None:
            raise ValueError("CSV data not loaded. Call validate_csv_structure() first.")
        
        # Group by project to get unique projects
        project_groups = self.data_df.groupby(['Project Code', 'Project Name', 'Project Type']).first()
        
        projects = []
        for (project_code, project_name, project_type), _ in project_groups.iterrows():
            # Determine project manager from allocations (first Manager role or first Tech Lead)
            project_allocations = self.data_df[self.data_df['Project Code'] == project_code]
            manager_row = project_allocations[project_allocations['Role'] == 'Manager']
            
            if manager_row.empty:
                manager_row = project_allocations[project_allocations['Role'] == 'Tech Lead']
            
            manager_id = None
            if not manager_row.empty:
                manager_name = manager_row.iloc[0]['Name']
                manager_id = self.get_employee_code_for_name(manager_name)
            
            # Determine project dates from allocations using robust parsing
            parsed_dates = []
            for date_val in project_allocations['Available From']:
                parsed_date = self._parse_date_flexible(date_val)
                if parsed_date is not None:
                    parsed_dates.append(parsed_date)
            
            if parsed_dates:
                start_date = min(parsed_dates).date()
            else:
                # Fallback to current date if no valid dates found
                start_date = date.today()
                logger.warning(f"No valid dates found for project {project_code}, using current date")
            
            projects.append({
                'project_id': project_code,
                'project_name': project_name,
                'client_name': f'{project_type} Client',  # Placeholder - enhance based on business rules
                'status': 'Active',
                'start_date': start_date,
                'end_date': None,  # Will be set when project completes
                'manager_id': manager_id
            })
        
        logger.info(f"Extracted {len(projects)} unique projects")
        return projects
    
    def prepare_allocations(self) -> List[Dict]:
        """
        Prepare allocation records for database insertion.
        
        Returns:
            List of allocation dictionaries
        """
        if self.data_df is None:
            raise ValueError("CSV data not loaded. Call validate_csv_structure() first.")
        
        allocations = []
        for _, row in self.data_df.iterrows():
            # Get employee code using proper mapping
            employee_code = self.get_employee_code_for_name(row['Name'])
            if not employee_code:
                logger.warning(f"Skipping allocation for '{row['Name']}' - employee not found")
                continue
                
            try:
                # Use robust parsing for allocation percentage
                allocation_percentage = self._parse_allocation_percentage(row['% Allocation'])
                if allocation_percentage is None:
                    logger.warning(f"Skipping allocation for {row['Name']} - invalid allocation percentage: {row['% Allocation']}")
                    continue
                
                # Use robust parsing for date
                effective_from_parsed = self._parse_date_flexible(row['Available From'])
                if effective_from_parsed is None:
                    logger.warning(f"Skipping allocation for {row['Name']} - invalid date: {row['Available From']}")
                    continue
                
                effective_from = effective_from_parsed.date()
                
                allocation = {
                    'employee_code': employee_code,
                    'project_id': row['Project Code'],
                    'allocation_percentage': allocation_percentage,
                    'effective_from': effective_from,
                    'effective_to': None,  # Open-ended allocation
                    'status': 'Active',
                    'created_by': 'SYSTEM_SEED',
                    'change_reason': f'Initial allocation - {row.get("Comments", "")}'
                }
                allocations.append(allocation)
                
            except Exception as e:
                logger.warning(f"Skipping allocation for {row['Name']} on {row['Project Code']}: {e}")
        
        logger.info(f"Prepared {len(allocations)} allocation records")
        return allocations
    
    def seed_projects(self, projects: List[Dict]) -> bool:
        """
        Insert project records into the database.
        
        Args:
            projects: List of project dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with get_cursor() as cursor:
                # Check existing projects to avoid duplicates
                cursor.execute("SELECT project_id FROM project")
                existing_projects = {row[0] for row in cursor.fetchall()}
                
                # Filter out existing projects
                new_projects = [p for p in projects if p['project_id'] not in existing_projects]
                
                if not new_projects:
                    logger.info("No new projects to insert")
                    return True
                
                # Prepare bulk insert
                insert_query = """
                    INSERT INTO project (
                        project_id, project_name, client_name, status, 
                        start_date, end_date, manager_id, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                project_data = [
                    (
                        p['project_id'], p['project_name'], p['client_name'], 
                        p['status'], p['start_date'], p['end_date'], 
                        p['manager_id'], datetime.now()
                    )
                    for p in new_projects
                ]
                
                cursor.executemany(insert_query, project_data)
                logger.info(f"Successfully inserted {len(new_projects)} projects")
                
                # Log the inserted projects
                for project in new_projects:
                    logger.info(f"  → {project['project_id']}: {project['project_name']}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error seeding projects: {e}")
            return False
    
    def seed_allocations(self, allocations: List[Dict]) -> bool:
        """
        Insert allocation records into the database.
        
        Args:
            allocations: List of allocation dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with get_cursor() as cursor:
                # Validate that referenced projects and employees exist
                cursor.execute("SELECT project_id FROM project")
                existing_projects = {row[0] for row in cursor.fetchall()}
                
                cursor.execute("SELECT employee_code FROM employee")
                existing_employees = {row[0] for row in cursor.fetchall()}
                
                # Filter valid allocations
                valid_allocations = []
                for allocation in allocations:
                    if allocation['project_id'] not in existing_projects:
                        logger.warning(f"Project {allocation['project_id']} not found, skipping allocation")
                        continue
                    if allocation['employee_code'] not in existing_employees:
                        logger.warning(f"Employee {allocation['employee_code']} not found, skipping allocation")
                        continue
                    valid_allocations.append(allocation)
                
                if not valid_allocations:
                    logger.warning("No valid allocations to insert")
                    return True
                
                # Check for existing allocations to avoid duplicates
                cursor.execute("""
                    SELECT employee_code, project_id, effective_from 
                    FROM project_allocation 
                    WHERE status = 'Active'
                """)
                existing_allocations = {
                    (row[0], row[1], row[2]) for row in cursor.fetchall()
                }
                
                # Filter out duplicates
                new_allocations = [
                    a for a in valid_allocations 
                    if (a['employee_code'], a['project_id'], a['effective_from']) not in existing_allocations
                ]
                
                if not new_allocations:
                    logger.info("No new allocations to insert")
                    return True
                
                # Prepare bulk insert
                insert_query = """
                    INSERT INTO project_allocation (
                        employee_code, project_id, allocation_percentage,
                        effective_from, effective_to, status, created_by,
                        created_at, change_reason
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                allocation_data = [
                    (
                        a['employee_code'], a['project_id'], a['allocation_percentage'],
                        a['effective_from'], a['effective_to'], a['status'],
                        a['created_by'], datetime.now(), a['change_reason']
                    )
                    for a in new_allocations
                ]
                
                cursor.executemany(insert_query, allocation_data)
                logger.info(f"Successfully inserted {len(new_allocations)} allocations")
                
                # Log allocation summary
                allocation_summary = {}
                for allocation in new_allocations:
                    key = f"{allocation['employee_code']} → {allocation['project_id']}"
                    allocation_summary[key] = allocation['allocation_percentage']
                
                logger.info("Allocation Summary:")
                for key, percentage in allocation_summary.items():
                    logger.info(f"  → {key}: {percentage}%")
                
                return True
                
        except Exception as e:
            logger.error(f"Error seeding allocations: {e}")
            return False
    
    def run_seeding(self) -> bool:
        """
        Execute the complete seeding process.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STARTING PROJECT AND ALLOCATION DATA SEEDING")
        logger.info("=" * 80)
        
        try:
            # Step 0: Load employee mapping from database
            logger.info("Step 0: Loading employee mapping from database...")
            if not self.load_employee_mapping():
                logger.error("Failed to load employee mapping. Aborting seeding process.")
                return False
            
            # Step 1: Validate CSV structure
            logger.info("Step 1: Validating CSV structure...")
            if not self.validate_csv_structure():
                logger.error("CSV validation failed. Aborting seeding process.")
                return False
            
            # Step 2: Extract projects
            logger.info("Step 2: Extracting unique projects...")
            projects = self.extract_unique_projects()
            
            # Step 3: Prepare allocations
            logger.info("Step 3: Preparing allocation data...")
            allocations = self.prepare_allocations()
            
            # Step 4: Seed projects
            logger.info("Step 4: Seeding projects table...")
            if not self.seed_projects(projects):
                logger.error("Project seeding failed. Aborting.")
                return False
            
            # Step 5: Seed allocations
            logger.info("Step 5: Seeding allocations table...")
            if not self.seed_allocations(allocations):
                logger.error("Allocation seeding failed.")
                return False
            
            logger.info("=" * 80)
            logger.info("DATA SEEDING COMPLETED SUCCESSFULLY!")
            logger.info(f"Projects seeded: {len(projects)}")
            logger.info(f"Allocations seeded: {len(allocations)}")
            
            # Generate detailed report
            self._generate_seeding_report(projects, allocations)
            
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error during seeding process: {e}")
            return False

    def _generate_seeding_report(self, projects: List[Dict], allocations: List[Dict]) -> None:
        """
        Generate detailed seeding report including parsing statistics.
        
        Args:
            projects: List of seeded projects
            allocations: List of seeded allocations
        """
        logger.info("=" * 60)
        logger.info("SEEDING REPORT")
        logger.info("=" * 60)
        
        # Basic statistics
        logger.info(f"📊 SUMMARY STATISTICS:")
        logger.info(f"  • Total projects seeded: {len(projects)}")
        logger.info(f"  • Total allocations seeded: {len(allocations)}")
        
        # Parsing statistics
        logger.info(f"\n🔍 DATA PARSING STATISTICS:")
        logger.info(f"  • Allocation percentage formats found: {len(self.parsing_stats['allocation_formats_found'])}")
        logger.info(f"  • Date formats found: {len(self.parsing_stats['date_formats_found'])}")
        logger.info(f"  • Successful allocation parses: {self.parsing_stats['successful_allocation_parses']}")
        logger.info(f"  • Successful date parses: {self.parsing_stats['successful_date_parses']}")
        logger.info(f"  • Allocation parsing failures: {len(self.parsing_stats['allocation_parsing_failures'])}")
        logger.info(f"  • Date parsing failures: {len(self.parsing_stats['date_parsing_failures'])}")
        
        # Show detected formats
        if self.parsing_stats['allocation_formats_found']:
            logger.info(f"\n📋 ALLOCATION FORMATS DETECTED:")
            for fmt in sorted(self.parsing_stats['allocation_formats_found']):
                logger.info(f"  • '{fmt}'")
        
        if self.parsing_stats['date_formats_found']:
            logger.info(f"\n📅 DATE FORMATS DETECTED:")
            for fmt in sorted(self.parsing_stats['date_formats_found']):
                logger.info(f"  • '{fmt}'")
        
        # Show parsing failures if any
        if self.parsing_stats['allocation_parsing_failures']:
            logger.info(f"\n⚠️  ALLOCATION PARSING FAILURES:")
            for failure in self.parsing_stats['allocation_parsing_failures'][:5]:  # Show first 5
                logger.info(f"  • {failure}")
            if len(self.parsing_stats['allocation_parsing_failures']) > 5:
                logger.info(f"  • ... and {len(self.parsing_stats['allocation_parsing_failures']) - 5} more")
        
        if self.parsing_stats['date_parsing_failures']:
            logger.info(f"\n⚠️  DATE PARSING FAILURES:")
            for failure in self.parsing_stats['date_parsing_failures'][:5]:  # Show first 5
                logger.info(f"  • {failure}")
            if len(self.parsing_stats['date_parsing_failures']) > 5:
                logger.info(f"  • ... and {len(self.parsing_stats['date_parsing_failures']) - 5} more")
        
        # Project summary
        if projects:
            project_types = {}
            for project in projects:
                # Extract project type from client name (contains project type)
                client_name = project.get('client_name', 'Unknown')
                project_type = client_name.replace(' Client', '') if ' Client' in client_name else 'Unknown'
                project_types[project_type] = project_types.get(project_type, 0) + 1
            
            logger.info(f"\n🎯 PROJECTS BY TYPE:")
            for proj_type, count in sorted(project_types.items()):
                logger.info(f"  • {proj_type}: {count} projects")
        
        # Allocation summary
        if allocations:
            allocation_summary = {}
            total_allocation = 0
            for allocation in allocations:
                emp_code = allocation['employee_code']
                percentage = allocation['allocation_percentage']
                allocation_summary[emp_code] = allocation_summary.get(emp_code, 0) + percentage
                total_allocation += percentage
            
            logger.info(f"\n👥 ALLOCATION SUMMARY:")
            logger.info(f"  • Employees with allocations: {len(allocation_summary)}")
            logger.info(f"  • Average allocation per employee: {total_allocation / len(allocation_summary):.1f}%")
            
            # Show over-allocated employees
            over_allocated = [emp for emp, alloc in allocation_summary.items() if alloc > 100]
            if over_allocated:
                logger.info(f"  • Over-allocated employees: {len(over_allocated)}")
                for emp in over_allocated[:3]:  # Show first 3
                    logger.info(f"    - {emp}: {allocation_summary[emp]:.1f}%")
        
        logger.info("=" * 60)


def verify_employee_data() -> bool:
    """
    Verify that required employees exist in the database.
    If not, create realistic employee records based on CSV data.
    """
    logger.info("Verifying employee data...")
    
    # Read CSV to get actual employee names that need to exist
    csv_file_path = Path("allocation/allocations.csv")
    if not csv_file_path.exists():
        logger.error(f"CSV file not found: {csv_file_path}")
        return False
    
    try:
        df = pd.read_csv(csv_file_path)
        csv_employee_names = df['Name'].unique().tolist()
        logger.info(f"Found {len(csv_employee_names)} unique employees in CSV: {csv_employee_names}")
        
        with get_cursor() as cursor:
            cursor.execute("SELECT employee_code, employee_name FROM employee WHERE status = 'Active'")
            existing_employees = {row[1]: row[0] for row in cursor.fetchall()}
            
            logger.info(f"Found {len(existing_employees)} existing employees in database")
            
            # Find missing employees using normalized name matching
            missing_employees = []
            seeder = ProjectDataSeeder()  # Temporary instance for name normalization
            
            for csv_name in csv_employee_names:
                normalized_csv_name = seeder._normalize_employee_name(csv_name)
                
                # Check if employee exists with exact or normalized name
                found = False
                for db_name in existing_employees.keys():
                    normalized_db_name = seeder._normalize_employee_name(db_name)
                    if (csv_name == db_name or  # Exact match
                        normalized_csv_name == normalized_db_name or  # Normalized match
                        csv_name.lower() == db_name.lower()):  # Case-insensitive match
                        found = True
                        logger.debug(f"Employee '{csv_name}' matched with database entry '{db_name}'")
                        break
                
                if not found:
                    logger.warning(f"Employee '{csv_name}' not found in database")
                    missing_employees.append(csv_name)
                else:
                    logger.debug(f"Employee '{csv_name}' found in database")
            
            if missing_employees:
                logger.warning(f"Missing {len(missing_employees)} required employees. Creating records...")
                
                # Get existing emails and employee codes to avoid conflicts
                cursor.execute("SELECT email FROM employee WHERE email IS NOT NULL")
                existing_emails = {row[0] for row in cursor.fetchall()}
                
                cursor.execute("SELECT employee_code FROM employee")
                existing_codes = {row[0] for row in cursor.fetchall()}
                
                # Generate realistic employee data
                employee_data = []
                for i, emp_name in enumerate(missing_employees, start=1):
                    # Generate unique employee code
                    base_code = f"EMP{1000 + len(existing_employees) + i:03d}"
                    emp_code = base_code
                    counter = 1
                    while emp_code in existing_codes:
                        emp_code = f"EMP{1000 + len(existing_employees) + i + counter:03d}"
                        counter += 1
                    existing_codes.add(emp_code)
                    
                    # Generate unique email
                    email_name = emp_name.lower().replace(' ', '.')
                    base_email = f"{email_name}@company.com"
                    email = base_email
                    counter = 1
                    while email in existing_emails:
                        email = f"{email_name}{counter}@company.com"
                        counter += 1
                    existing_emails.add(email)
                    
                    # Determine employee type and grade based on role in CSV
                    emp_roles = df[df['Name'] == emp_name]['Role'].unique()
                    if 'Manager' in emp_roles:
                        emp_type = 'Full-time'
                        grade = 'L3'
                    elif 'Tech Lead' in emp_roles:
                        emp_type = 'Full-time' 
                        grade = 'L2'
                    elif 'Senior Developer' in emp_roles:
                        emp_type = 'Full-time'
                        grade = 'L2'
                    else:
                        emp_type = 'Full-time'
                        grade = 'L1'
                    
                    employee_data.append((
                        emp_code,
                        emp_name,
                        email,
                        date(2023, 1, 1),  # Dummy join date
                        emp_type,
                        'Active',
                        grade,
                        datetime.now()
                    ))
                
                # Insert employees
                insert_query = """
                    INSERT INTO employee (
                        employee_code, employee_name, email, date_of_joining,
                        employee_type, status, grade, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO NOTHING
                """
                
                cursor.executemany(insert_query, employee_data)
                logger.info(f"Created {len(missing_employees)} employee records:")
                
                for emp_data in employee_data:
                    logger.info(f"  → {emp_data[0]}: {emp_data[1]} ({emp_data[6]}) - {emp_data[2]}")
            else:
                logger.info("All required employees already exist in database")
            
            return True
            
    except Exception as e:
        logger.error(f"Error verifying employee data: {e}")
        return False


def check_employee_name_conflicts() -> bool:
    """
    Check for potential employee name conflicts before running the seeder.
    
    Returns:
        bool: True if no conflicts, False if conflicts found
    """
    logger.info("Checking for employee name conflicts...")
    
    try:
        with get_cursor() as cursor:
            # Find duplicate normalized names
            cursor.execute("""
                SELECT LOWER(TRIM(employee_name)) as normalized_name, COUNT(*) as count
                FROM employee 
                WHERE status = 'Active'
                GROUP BY LOWER(TRIM(employee_name))
                HAVING COUNT(*) > 1
                ORDER BY count DESC
            """)
            
            conflicts = cursor.fetchall()
            
            if conflicts:
                logger.error(f"Found {len(conflicts)} employee name conflicts:")
                for normalized_name, count in conflicts:
                    logger.error(f"  → '{normalized_name}' appears {count} times")
                    
                    # Get the specific employees with this conflict
                    cursor.execute("""
                        SELECT employee_code, employee_name, email
                        FROM employee 
                        WHERE LOWER(TRIM(employee_name)) = LOWER(TRIM(%s))
                        AND status = 'Active'
                    """, (normalized_name,))
                    
                    conflicted_employees = cursor.fetchall()
                    for emp_code, emp_name, email in conflicted_employees:
                        logger.error(f"    • {emp_code}: {emp_name} ({email})")
                
                logger.error("Please resolve name conflicts before running the seeder.")
                logger.error("Suggestions:")
                logger.error("1. Update employee names to be unique (e.g., add middle initial)")
                logger.error("2. Mark duplicate employees as 'Inactive'")
                logger.error("3. Update the CSV to use employee codes instead of names")
                
                return False
            else:
                logger.info("No employee name conflicts found")
                return True
                
    except Exception as e:
        logger.error(f"Error checking employee name conflicts: {e}")
        return False


def validate_seeding_results() -> bool:
    """
    Validate the results of the seeding operation.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating seeding results...")
    
    try:
        with get_cursor() as cursor:
            # Check project count
            cursor.execute("SELECT COUNT(*) FROM project WHERE status = 'Active'")
            project_count = cursor.fetchone()[0]
            
            # Check allocation count
            cursor.execute("SELECT COUNT(*) FROM project_allocation WHERE status = 'Active'")
            allocation_count = cursor.fetchone()[0]
            
            # Check for orphaned allocations (allocations without matching projects)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM project_allocation pa 
                LEFT JOIN project p ON pa.project_id = p.project_id 
                WHERE p.project_id IS NULL AND pa.status = 'Active'
            """)
            orphaned_allocations = cursor.fetchone()[0]
            
            # Check for allocations with invalid employees
            cursor.execute("""
                SELECT COUNT(*) 
                FROM project_allocation pa 
                LEFT JOIN employee e ON pa.employee_code = e.employee_code 
                WHERE e.employee_code IS NULL AND pa.status = 'Active'
            """)
            invalid_employee_allocations = cursor.fetchone()[0]
            
            # Check employee over-allocation
            cursor.execute("""
                SELECT employee_code, SUM(allocation_percentage) as total_allocation
                FROM project_allocation 
                WHERE status = 'Active' 
                GROUP BY employee_code 
                HAVING SUM(allocation_percentage) > 100
            """)
            over_allocated_employees = cursor.fetchall()
            
            # Report results
            logger.info(f"✓ Active projects: {project_count}")
            logger.info(f"✓ Active allocations: {allocation_count}")
            
            validation_passed = True
            
            if orphaned_allocations > 0:
                logger.error(f"❌ Found {orphaned_allocations} orphaned allocations (no matching project)")
                validation_passed = False
            
            if invalid_employee_allocations > 0:
                logger.error(f"❌ Found {invalid_employee_allocations} allocations with invalid employees")
                validation_passed = False
            
            if over_allocated_employees:
                logger.warning(f"⚠️  Found {len(over_allocated_employees)} over-allocated employees:")
                for emp_code, total_alloc in over_allocated_employees:
                    logger.warning(f"  → {emp_code}: {total_alloc}%")
            
            if validation_passed:
                logger.info("🎉 All validation checks passed!")
            else:
                logger.error("❌ Validation failed - please review the issues above")
            
            return validation_passed
            
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return False


def main():
    """
    Main execution function with comprehensive error handling and reporting.
    """
    try:
        logger.info("Project Data Seeder - Enterprise Edition")
        logger.info(f"Timestamp: {datetime.now()}")
        logger.info(f"Database: {db_config.host}:{db_config.port}/{db_config.database}")
        
        # Check for employee name conflicts first
        if not check_employee_name_conflicts():
            logger.error("Employee name conflicts detected. Cannot proceed safely.")
            sys.exit(1)
        
        # Verify employee data 
        if not verify_employee_data():
            logger.error("Employee data verification failed. Cannot proceed.")
            sys.exit(1)
        
        # Initialize and run seeder
        seeder = ProjectDataSeeder()
        success = seeder.run_seeding()
        
        if success:
            logger.info("🎉 Data seeding completed successfully!")
            
            # Validate seeding results
            if validate_seeding_results():
                logger.info("Seeding results validation passed.")
            else:
                logger.error("Seeding results validation failed. Please review the issues.")
            
            sys.exit(0)
        else:
            logger.error("❌ Data seeding failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()