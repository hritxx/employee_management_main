import streamlit as st
import pandas as pd
from logs.activity_logger import get_logger

def render_custom_queries(engine):
    """Render the custom queries interface"""
    st.header("Custom Query Builder")
    
    # Get activity logger
    activity_logger = get_logger(engine)

    # Cache the filter options to avoid repeated queries
    @st.cache_data
    def get_filter_options():
        try:
            # Get list of employee names with null handling
            employee_query = """
                SELECT DISTINCT employee_name 
                FROM employee 
                WHERE employee_name IS NOT NULL AND employee_name != ''
                ORDER BY employee_name
            """
            employee_names_df = pd.read_sql(employee_query, engine)
            employee_names = employee_names_df["employee_name"].tolist()

            # Get list of departments with null handling
            dept_query = """
                SELECT DISTINCT d.department_name 
                FROM department d
                JOIN employee e ON e.department_id = d.department_id
                WHERE d.department_name IS NOT NULL AND d.department_name != ''
                ORDER BY d.department_name
            """
            departments_df = pd.read_sql(dept_query, engine)
            departments = departments_df["department_name"].tolist()

            # Get list of projects with null handling
            proj_query = """
                SELECT DISTINCT project_name 
                FROM project 
                WHERE project_name IS NOT NULL AND project_name != ''
                ORDER BY project_name
            """
            projects_df = pd.read_sql(proj_query, engine)
            projects = projects_df["project_name"].tolist()

            return employee_names, departments, projects

        except Exception as e:
            st.error(f"Error loading filter options: {e}")
            
            # Log error
            activity_logger.log_event(
                event_type="ERROR",
                description=f"Error loading filter options: {str(e)}",
                user=st.session_state.get('username', 'unknown')
            )
            
            return [], [], []

    employee_names, departments, projects = get_filter_options()

    # Create filter columns
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_employees = st.multiselect("Select Employees", options=["All"] + employee_names, default=["All"])

    with col2:
        selected_departments = st.multiselect("Select Departments", options=["All"] + departments, default=["All"])

    with col3:
        selected_projects = st.multiselect("Select Projects", options=["All"] + projects, default=["All"])

    # Create date range filter and employee status filter
    col4, col5, col6 = st.columns(3)
    with col4:
        start_date = st.date_input("Start Date", value=None)
    with col5:
        end_date = st.date_input("End Date", value=None)
    with col6:
        employee_status = st.selectbox("Employee Status", options=["All", "Active", "Inactive"], index=0)

    # Report type selection for custom query
    report_type = st.selectbox(
        "Select Report Type",
        ["Employee Details", "Project Assignments", "Attendance Records", "Timesheet Summary"]
    )

    # Build the query
    if st.button("Generate Report", key="custom_query_report"):
        try:
            df = pd.DataFrame()  # Initialize empty dataframe
            query = ""  # Initialize query string
            params = []  # Initialize params list

            if report_type == "Employee Details":
                query = """
                    SELECT 
                        e.employee_code,
                        e.employee_name,
                        d.department_name as department,
                        des.designation_name as designation,
                        e.email,
                        e.date_of_joining,
                        e.employee_type,
                        e.grade,
                        e.status,
                        e.current_experience,
                        e.past_experience,
                        e.total_experience
                    FROM employee e
                    LEFT JOIN department d ON e.department_id = d.department_id
                    LEFT JOIN designation des ON e.designation_id = des.designation_id
                    WHERE 1=1
                """

                # Add employee status filter
                if employee_status == "Active":
                    query += " AND (e.status = 'Active' OR e.status IS NULL OR UPPER(e.status) IN ('ACTIVE', 'A'))"
                elif employee_status == "Inactive":
                    query += " AND (e.status IS NOT NULL AND UPPER(e.status) IN ('INACTIVE', 'I', 'TERMINATED', 'RESIGNED', 'DISABLED'))"

                if "All" not in selected_employees and selected_employees:
                    placeholders = ",".join(["%s" for _ in selected_employees])
                    query += f" AND e.employee_name IN ({placeholders})"
                    params.extend(selected_employees)

                if "All" not in selected_departments and selected_departments:
                    placeholders = ",".join(["%s" for _ in selected_departments])
                    query += f" AND d.department_name IN ({placeholders})"
                    params.extend(selected_departments)

                df = pd.read_sql(query, engine, params=tuple(params))

            elif report_type == "Project Assignments":
                query = """
                    SELECT 
                        t.work_date as date,
                        t.employee_code,
                        e.employee_name,
                        d.department_name as department,
                        p.project_id,
                        p.project_name,
                        t.hours_worked
                    FROM timesheet t
                    JOIN employee e ON t.employee_code = e.employee_code
                    LEFT JOIN department d ON e.department_id = d.department_id
                    JOIN project p ON t.project_id = p.project_id
                    WHERE 1=1
                """

                # Add employee status filter
                if employee_status == "Active":
                    query += " AND (e.status = 'Active' OR e.status IS NULL OR UPPER(e.status) IN ('ACTIVE', 'A'))"
                elif employee_status == "Inactive":
                    query += " AND (e.status IS NOT NULL AND UPPER(e.status) IN ('INACTIVE', 'I', 'TERMINATED', 'RESIGNED', 'DISABLED'))"

                if "All" not in selected_employees and selected_employees:
                    placeholders = ",".join(["%s" for _ in selected_employees])
                    query += f" AND e.employee_name IN ({placeholders})"
                    params.extend(selected_employees)

                if "All" not in selected_departments and selected_departments:
                    placeholders = ",".join(["%s" for _ in selected_departments])
                    query += f" AND d.department_name IN ({placeholders})"
                    params.extend(selected_departments)

                if "All" not in selected_projects and selected_projects:
                    placeholders = ",".join(["%s" for _ in selected_projects])
                    query += f" AND p.project_name IN ({placeholders})"
                    params.extend(selected_projects)

                if start_date:
                    query += " AND t.work_date >= %s"
                    params.append(start_date.strftime('%Y-%m-%d'))

                if end_date:
                    query += " AND t.work_date <= %s"
                    params.append(end_date.strftime('%Y-%m-%d'))

                query += " ORDER BY t.work_date, e.employee_name"

                df = pd.read_sql(query, engine, params=tuple(params))

            elif report_type == "Attendance Records":
                query = """
                    SELECT 
                        a.attendance_date as date,
                        a.employee_code, 
                        e.employee_name,
                        d.department_name as department,
                        a.clock_in_time,
                        a.clock_out_time,
                        a.total_hours,
                        a.attendance_type
                    FROM attendance a
                    JOIN employee e ON a.employee_code = e.employee_code
                    LEFT JOIN department d ON e.department_id = d.department_id
                    WHERE 1=1
                """

                # Add employee status filter
                if employee_status == "Active":
                    query += " AND (e.status = 'Active' OR e.status IS NULL OR UPPER(e.status) IN ('ACTIVE', 'A'))"
                elif employee_status == "Inactive":
                    query += " AND (e.status IS NOT NULL AND UPPER(e.status) IN ('INACTIVE', 'I', 'TERMINATED', 'RESIGNED', 'DISABLED'))"

                if "All" not in selected_employees and selected_employees:
                    placeholders = ",".join(["%s" for _ in selected_employees])
                    query += f" AND e.employee_name IN ({placeholders})"
                    params.extend(selected_employees)

                if "All" not in selected_departments and selected_departments:
                    placeholders = ",".join(["%s" for _ in selected_departments])
                    query += f" AND d.department_name IN ({placeholders})"
                    params.extend(selected_departments)

                if start_date:
                    query += " AND a.attendance_date >= %s"
                    params.append(start_date.strftime('%Y-%m-%d'))

                if end_date:
                    query += " AND a.attendance_date <= %s"
                    params.append(end_date.strftime('%Y-%m-%d'))

                query += " ORDER BY a.attendance_date, e.employee_name"

                df = pd.read_sql(query, engine, params=tuple(params))

            elif report_type == "Timesheet Summary":
                query = """
                    SELECT 
                        e.employee_code,
                        e.employee_name,
                        d.department_name as department,
                        p.project_id,
                        p.project_name,
                        SUM(t.hours_worked) as total_hours,
                        COUNT(DISTINCT t.work_date) as days_worked,
                        MIN(t.work_date) as first_day,
                        MAX(t.work_date) as last_day
                    FROM timesheet t
                    JOIN employee e ON t.employee_code = e.employee_code
                    LEFT JOIN department d ON e.department_id = d.department_id
                    JOIN project p ON t.project_id = p.project_id
                    WHERE 1=1
                """

                # Add employee status filter
                if employee_status == "Active":
                    query += " AND (e.status = 'Active' OR e.status IS NULL OR UPPER(e.status) IN ('ACTIVE', 'A'))"
                elif employee_status == "Inactive":
                    query += " AND (e.status IS NOT NULL AND UPPER(e.status) IN ('INACTIVE', 'I', 'TERMINATED', 'RESIGNED', 'DISABLED'))"

                if "All" not in selected_employees and selected_employees:
                    placeholders = ",".join(["%s" for _ in selected_employees])
                    query += f" AND e.employee_name IN ({placeholders})"
                    params.extend(selected_employees)

                if "All" not in selected_departments and selected_departments:
                    placeholders = ",".join(["%s" for _ in selected_departments])
                    query += f" AND d.department_name IN ({placeholders})"
                    params.extend(selected_departments)

                if "All" not in selected_projects and selected_projects:
                    placeholders = ",".join(["%s" for _ in selected_projects])
                    query += f" AND p.project_name IN ({placeholders})"
                    params.extend(selected_projects)

                if start_date:
                    query += " AND t.work_date >= %s"
                    params.append(start_date.strftime('%Y-%m-%d'))

                if end_date:
                    query += " AND t.work_date <= %s"
                    params.append(end_date.strftime('%Y-%m-%d'))

                query += " GROUP BY e.employee_code, e.employee_name, d.department_name, p.project_id, p.project_name"
                query += " ORDER BY e.employee_name, p.project_name"

                df = pd.read_sql(query, engine, params=tuple(params))

            # Log the query
            query_details = {
                "report_type": report_type,
                "filters": {
                    "employees": selected_employees,
                    "departments": selected_departments,
                    "projects": selected_projects,
                    "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                    "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
                    "employee_status": employee_status
                }
            }
            
            # Log the query execution
            activity_logger.log_query(
                query_text=query,
                user=st.session_state.get('username', 'unknown'),
                query_type="CUSTOM",
                status="SUCCESS"
            )

            # Display results
            if not df.empty:
                st.subheader(f"{report_type} Report")
                st.dataframe(df, use_container_width=True)
                
                # Add download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{report_type.lower().replace(' ', '_')}_report.csv",
                    mime="text/csv",
                )
            else:
                st.info("No data found matching your criteria")
                
                # Log empty result
                activity_logger.log_event(
                    event_type="QUERY_RESULT",
                    description=f"Custom query returned no results: {report_type}",
                    user=st.session_state.get('username', 'unknown'),
                    details=query_details
                )

        except Exception as e:
            st.error(f"Error executing query: {e}")
            
            # Log error
            activity_logger.log_event(
                event_type="ERROR",
                description=f"Custom query error: {str(e)}",
                user=st.session_state.get('username', 'unknown'),
                details={"report_type": report_type}
            )