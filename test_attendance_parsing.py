#!/usr/bin/env python3

import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_safe_value(row, column_name: str, default_value=None):
    """Safely get value from DataFrame row, handling missing columns"""
    try:
        if column_name in row.index and pd.notna(row[column_name]):
            return row[column_name]
        return default_value
    except Exception:
        return default_value

def test_attendance_parsing():
    """Test attendance data parsing"""
    try:
        # Load the attendance CSV
        df_attendance = pd.read_csv('updated_csv_files/attendance-report.csv')
        logger.info(f"Loaded {len(df_attendance)} attendance records")
        
        print("CSV Columns:", df_attendance.columns.tolist())
        print("\nFirst few rows:")
        print(df_attendance.head())
        
        # Test parsing logic
        attendance_data = []
        errors = []
        
        for idx, row in df_attendance.iterrows():
            try:
                # Use ShiftDate instead of Date
                shift_date = get_safe_value(row, 'ShiftDate')
                in_time = get_safe_value(row, 'In Time')
                out_time = get_safe_value(row, 'Out Time')
                status = get_safe_value(row, 'Status', 'Present')
                employee_code = get_safe_value(row, 'Employee Code')

                print(f"\nRow {idx + 1}:")
                print(f"  Employee Code: {employee_code}")
                print(f"  ShiftDate: {shift_date} (type: {type(shift_date)})")
                print(f"  In Time: {in_time} (type: {type(in_time)})")
                print(f"  Out Time: {out_time} (type: {type(out_time)})")
                print(f"  Status: {status}")

                if shift_date:
                    # Parse date
                    parsed_date = pd.to_datetime(shift_date).date()
                    print(f"  Parsed Date: {parsed_date}")
                    
                    # Parse times
                    parsed_in_time = pd.to_datetime(in_time, format='%H:%M:%S').time() if in_time else None
                    parsed_out_time = pd.to_datetime(out_time, format='%H:%M:%S').time() if out_time else None
                    
                    print(f"  Parsed In Time: {parsed_in_time}")
                    print(f"  Parsed Out Time: {parsed_out_time}")
                    
                    attendance_data.append((
                        parsed_date,
                        employee_code,
                        parsed_in_time,
                        parsed_out_time,
                        status
                    ))
                    print(f"  ✓ Successfully parsed row {idx + 1}")
                else:
                    print(f"  ✗ No shift date found for row {idx + 1}")
                    
            except Exception as e:
                error_msg = f"Failed to parse attendance record at row {idx + 1}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                print(f"  ✗ Error: {e}")

        print(f"\n=== SUMMARY ===")
        print(f"Total records: {len(df_attendance)}")
        print(f"Successfully parsed: {len(attendance_data)}")
        print(f"Errors: {len(errors)}")
        
        if errors:
            print("\nErrors encountered:")
            for error in errors:
                print(f"  - {error}")
        
        if attendance_data:
            print(f"\nSample parsed data:")
            for i, record in enumerate(attendance_data[:3]):
                print(f"  Record {i+1}: {record}")
                
        return len(errors) == 0
        
    except Exception as e:
        logger.error(f"Error testing attendance parsing: {e}")
        return False

if __name__ == "__main__":
    test_attendance_parsing()
