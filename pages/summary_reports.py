import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from core.database import get_cursor

def show_available_tables():
    """Helper function to show available tables"""
    try:
        with get_cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cursor.fetchall()
            
            if tables:
                st.subheader("Available Tables:")
                for table in tables:
                    st.write(f"- {table[0]}")
            else:
                st.write("No tables found")
                
    except Exception as e:
        st.error(f"Error fetching tables: {e}")

def get_exit_report():
    """Get employee exit report with tenure calculations"""
    try:
        with get_cursor() as cursor:
            query = """
            SELECT 
                ee.employee_code,
                e.employee_name,
                d.department_name,
                des.designation_name,
                e.date_of_joining,
                ee.exit_date,
                ee.last_working_date,
                ee.exit_reason,
                ee.exit_comments,
                EXTRACT(YEAR FROM AGE(ee.exit_date, e.date_of_joining)) * 12 + 
                EXTRACT(MONTH FROM AGE(ee.exit_date, e.date_of_joining)) as tenure_months,
                EXTRACT(YEAR FROM AGE(ee.exit_date, e.date_of_joining)) as tenure_years
            FROM employee_exit ee
            JOIN employee e ON ee.employee_code = e.employee_code
            LEFT JOIN department d ON e.department_id = d.department_id
            LEFT JOIN designation des ON e.designation_id = des.designation_id
            ORDER BY ee.exit_date DESC
            """
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
    except Exception as e:
        st.error(f"Error fetching exit report: {str(e)}")
        return pd.DataFrame()

def get_experience_report():
    """Get employee experience report with categorization"""
    try:
        with get_cursor() as cursor:
            query = """
            SELECT 
                e.employee_code,
                e.employee_name,
                d.department_name,
                des.designation_name,
                e.current_experience,
                e.past_experience,
                e.total_experience,
                CASE 
                    WHEN e.total_experience < 1 THEN 'Fresher'
                    WHEN e.total_experience >= 1 AND e.total_experience < 3 THEN 'Junior (1-3 years)'
                    WHEN e.total_experience >= 3 AND e.total_experience < 7 THEN 'Mid-level (3-7 years)'
                    WHEN e.total_experience >= 7 AND e.total_experience < 12 THEN 'Senior (7-12 years)'
                    ELSE 'Expert (12+ years)'
                END as experience_level,
                e.date_of_joining,
                e.status
            FROM employee e
            LEFT JOIN department d ON e.department_id = d.department_id
            LEFT JOIN designation des ON e.designation_id = des.designation_id
            WHERE e.status = 'Active'
            ORDER BY e.total_experience DESC
            """
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
    except Exception as e:
        st.error(f"Error fetching experience report: {str(e)}")
        return pd.DataFrame()

def get_work_profile_report():
    """Get employee work profile with project allocation details"""
    try:
        with get_cursor() as cursor:
            query = """
            WITH project_counts AS (
                SELECT 
                    pa.employee_code,
                    COUNT(DISTINCT pa.project_id) as total_projects,
                    COUNT(DISTINCT CASE WHEN pa.status = 'Active' THEN pa.project_id END) as active_projects,
                    SUM(CASE WHEN pa.status = 'Active' THEN pa.allocation_percentage ELSE 0 END) as total_allocation
                FROM project_allocation pa
                GROUP BY pa.employee_code
            ),
            timesheet_summary AS (
                SELECT 
                    t.employee_code,
                    SUM(t.hours_worked) as total_hours_logged,
                    COUNT(DISTINCT t.work_date) as days_logged,
                    AVG(t.hours_worked) as avg_daily_hours
                FROM timesheet t
                WHERE t.work_date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY t.employee_code
            ),
            work_profile_summary AS (
                SELECT 
                    wp.employee_code,
                    wp.role,
                    wp.primary_skills,
                    wp.secondary_skills,
                    wp.total_experience_years,
                    wp.relevant_experience_years,
                    wp.certifications
                FROM employee_work_profile wp
            )
            SELECT 
                e.employee_code,
                e.employee_name,
                e.department_name,
                e.grade,
                wp.role as current_role,
                COALESCE(pc.total_projects, 0) as total_projects,
                COALESCE(pc.active_projects, 0) as active_projects,
                COALESCE(pc.total_allocation, 0) as total_allocation_percentage,
                COALESCE(ts.total_hours_logged, 0) as hours_last_30_days,
                COALESCE(ts.days_logged, 0) as days_logged_last_30_days,
                COALESCE(ts.avg_daily_hours, 0) as avg_daily_hours,
                COALESCE(wp.total_experience_years, 0) as total_experience_years,
                COALESCE(wp.relevant_experience_years, 0) as relevant_experience_years,
                wp.primary_skills,
                wp.secondary_skills,
                wp.certifications,
                e.status
            FROM employee e
            LEFT JOIN work_profile_summary wp ON e.employee_code = wp.employee_code
            LEFT JOIN project_counts pc ON e.employee_code = pc.employee_code
            LEFT JOIN timesheet_summary ts ON e.employee_code = ts.employee_code
            WHERE e.status = 'Active'
            ORDER BY e.employee_name
            """
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
    except Exception as e:
        st.error(f"Error fetching work profile report: {str(e)}")
        return pd.DataFrame()

def get_attendance_report():
    """Get attendance report for last 30 days"""
    try:
        with get_cursor() as cursor:
            query = """
            WITH attendance_summary AS (
                SELECT 
                    a.employee_code,
                    COUNT(*) as total_days,
                    COUNT(CASE WHEN a.attendance_type = 'Present' THEN 1 END) as present_days,
                    COUNT(CASE WHEN a.attendance_type = 'Absent' THEN 1 END) as absent_days,
                    AVG(CASE WHEN a.total_hours IS NOT NULL THEN a.total_hours ELSE 0 END) as avg_hours_per_day,
                    SUM(CASE WHEN a.total_hours IS NOT NULL THEN a.total_hours ELSE 0 END) as total_hours_worked
                FROM attendance a
                WHERE a.attendance_date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY a.employee_code
            )
            SELECT 
                e.employee_code,
                e.employee_name,
                d.department_name,
                des.designation_name,
                COALESCE(ast.total_days, 0) as total_days,
                COALESCE(ast.present_days, 0) as present_days,
                COALESCE(ast.absent_days, 0) as absent_days,
                CASE 
                    WHEN ast.total_days > 0 THEN 
                        ROUND((ast.present_days::DECIMAL / ast.total_days::DECIMAL) * 100, 2)
                    ELSE 0 
                END as attendance_percentage,
                COALESCE(ROUND(ast.avg_hours_per_day, 2), 0) as avg_hours_per_day,
                COALESCE(ROUND(ast.total_hours_worked, 2), 0) as total_hours_worked,
                e.status
            FROM employee e
            LEFT JOIN department d ON e.department_id = d.department_id
            LEFT JOIN designation des ON e.designation_id = des.designation_id
            LEFT JOIN attendance_summary ast ON e.employee_code = ast.employee_code
            WHERE e.status = 'Active'
            ORDER BY attendance_percentage DESC, e.employee_name
            """
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
    except Exception as e:
        st.error(f"Error fetching attendance report: {str(e)}")
        return pd.DataFrame()

def get_department_summary():
    """Get department-wise summary statistics with accurate counts"""
    try:
        with get_cursor() as cursor:
            query = """
            WITH employee_stats AS (
                SELECT 
                    e.department_id,
                    COUNT(*) as total_employees,
                    COUNT(CASE WHEN e.status = 'Active' THEN 1 END) as active_employees,
                    ROUND(AVG(e.total_experience), 2) as avg_experience
                FROM employee e
                GROUP BY e.department_id
            ),
            project_stats AS (
                SELECT 
                    e.department_id,
                    COUNT(DISTINCT pa.project_id) as total_projects
                FROM employee e
                LEFT JOIN project_allocation pa ON e.employee_code = pa.employee_code 
                    AND pa.status = 'Active'
                GROUP BY e.department_id
            )
            SELECT 
                d.department_name,
                d.business_unit,
                d.parent_department,
                COALESCE(es.total_employees, 0) as total_employees,
                COALESCE(es.active_employees, 0) as active_employees,
                COALESCE(es.avg_experience, 0) as avg_experience,
                COALESCE(ps.total_projects, 0) as total_projects,
                d.status as department_status
            FROM department d
            LEFT JOIN employee_stats es ON d.department_id = es.department_id
            LEFT JOIN project_stats ps ON d.department_id = ps.department_id
            WHERE d.status = 'Active'
            ORDER BY total_employees DESC, d.department_name
            """
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
    except Exception as e:
        st.error(f"Error fetching department summary: {str(e)}")
        return pd.DataFrame()

def get_detailed_department_analysis():
    """Get more detailed department analysis with additional metrics"""
    try:
        with get_cursor() as cursor:
            query = """
            WITH dept_metrics AS (
                SELECT 
                    d.department_id,
                    d.department_name,
                    d.business_unit,
                    d.parent_department,
                    COUNT(e.employee_code) as total_employees,
                    COUNT(CASE WHEN e.status = 'Active' THEN 1 END) as active_employees,
                    COUNT(CASE WHEN e.status = 'Inactive' THEN 1 END) as inactive_employees,
                    ROUND(AVG(CASE WHEN e.status = 'Active' THEN e.total_experience END), 2) as avg_active_experience,
                    ROUND(MIN(CASE WHEN e.status = 'Active' THEN e.total_experience END), 2) as min_experience,
                    ROUND(MAX(CASE WHEN e.status = 'Active' THEN e.total_experience END), 2) as max_experience,
                    COUNT(DISTINCT CASE WHEN e.status = 'Active' THEN e.designation_id END) as unique_designations,
                    COUNT(DISTINCT CASE WHEN pa.status = 'Active' THEN pa.project_id END) as active_projects
                FROM department d
                LEFT JOIN employee e ON d.department_id = e.department_id
                LEFT JOIN project_allocation pa ON e.employee_code = pa.employee_code 
                    AND pa.status = 'Active'
                WHERE d.status = 'Active'
                GROUP BY d.department_id, d.department_name, d.business_unit, d.parent_department
            )
            SELECT 
                *,
                CASE 
                    WHEN total_employees > 0 
                    THEN ROUND((active_employees::DECIMAL / total_employees) * 100, 1) 
                    ELSE 0 
                END as active_employee_percentage,
                CASE 
                    WHEN active_employees > 0 
                    THEN ROUND(active_projects::DECIMAL / active_employees, 2) 
                    ELSE 0 
                END as projects_per_employee
            FROM dept_metrics
            ORDER BY total_employees DESC, department_name
            """
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
    except Exception as e:
        st.error(f"Error fetching detailed department analysis: {str(e)}")
        return pd.DataFrame()

def get_department_hierarchy():
    """Get department hierarchy information"""
    try:
        with get_cursor() as cursor:
            query = """
            WITH RECURSIVE dept_hierarchy AS (
                -- Base case: departments without parent
                SELECT 
                    department_id,
                    department_name,
                    business_unit,
                    parent_department,
                    0 as level,
                    department_name as hierarchy_path
                FROM department 
                WHERE parent_department IS NULL AND status = 'Active'
                
                UNION ALL
                
                -- Recursive case: departments with parent
                SELECT 
                    d.department_id,
                    d.department_name,
                    d.business_unit,
                    d.parent_department,
                    dh.level + 1,
                    dh.hierarchy_path || ' > ' || d.department_name
                FROM department d
                JOIN dept_hierarchy dh ON d.parent_department = dh.department_name
                WHERE d.status = 'Active'
            ),
            dept_with_employees AS (
                SELECT 
                    dh.*,
                    COUNT(e.employee_code) as employee_count,
                    COUNT(CASE WHEN e.status = 'Active' THEN 1 END) as active_employee_count
                FROM dept_hierarchy dh
                LEFT JOIN employee e ON dh.department_id = e.department_id
                GROUP BY dh.department_id, dh.department_name, dh.business_unit, 
                         dh.parent_department, dh.level, dh.hierarchy_path
            )
            SELECT * FROM dept_with_employees
            ORDER BY level, hierarchy_path
            """
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
    except Exception as e:
        st.error(f"Error fetching department hierarchy: {str(e)}")
        return pd.DataFrame()

def render_summary_reports():
    """Main function to render the Summary Reports tab"""
    st.header("Summary Reports")
    
    # # Show available tables for debugging
    # with st.expander("Debug: Available Tables"):
    #     show_available_tables()
    
    # Create tabs for different summary reports
    summary_tabs = st.tabs(["Exit Report", "Experience Report", "Work Profile", "Attendance Report"])
    
    
        
    with summary_tabs[0]:  # Exit Report
                st.subheader("Employee Exit Report")
                exit_data = get_exit_report()
                if not exit_data.empty:
                    st.dataframe(exit_data, use_container_width=True)
                    
                    # Exit summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Exits", len(exit_data))
                    with col2:
                        avg_tenure = exit_data['tenure_months'].mean()
                        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
                    with col3:
                        # Convert exit_date to datetime if it's not already
                        exit_data['exit_date'] = pd.to_datetime(exit_data['exit_date'])
                        recent_exits = len(exit_data[exit_data['exit_date'] >= pd.Timestamp.now() - pd.DateOffset(months=3)])
                        st.metric("Recent Exits (3m)", recent_exits)
                else:
                    st.info("No exit data available.")
                    
    with summary_tabs[1]:  # Experience Report
        st.subheader("Experience Report")
        exp_data = get_experience_report()
        if not exp_data.empty:
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                dept_filter = st.multiselect("Filter by Department", 
                                           options=exp_data['department_name'].unique(),
                                           default=exp_data['department_name'].unique())
            with col2:
                exp_filter = st.multiselect("Filter by Experience Level",
                                          options=exp_data['experience_level'].unique(),
                                          default=exp_data['experience_level'].unique())
            
            # Apply filters
            filtered_exp_data = exp_data[
                (exp_data['department_name'].isin(dept_filter)) &
                (exp_data['experience_level'].isin(exp_filter))
            ]
            
            st.dataframe(filtered_exp_data, use_container_width=True)
            
            # Experience level distribution
            exp_level_dist = filtered_exp_data['experience_level'].value_counts()
            fig = px.pie(values=exp_level_dist.values, names=exp_level_dist.index,
                        title="Experience Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Department-wise experience
            dept_exp = filtered_exp_data.groupby('department_name')['total_experience'].mean().sort_values(ascending=False)
            
            # Create DataFrame for plotly express
            dept_exp_df = pd.DataFrame({
                'department': dept_exp.index,
                'avg_experience': dept_exp.values
            })
            
            fig2 = px.bar(dept_exp_df, x='department', y='avg_experience',
                         title="Average Experience by Department",
                         labels={'department': 'Department', 'avg_experience': 'Average Experience (Years)'})
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No experience data available.")
    
    with summary_tabs[2]:  # Work Profile
        st.subheader("Employee Work Profile")
        work_data = get_work_profile_report()
        if not work_data.empty:
            st.dataframe(work_data, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_projects = work_data['total_projects'].mean()
                st.metric("Avg Projects/Employee", f"{avg_projects:.1f}")
            with col2:
                avg_hours = work_data['hours_last_30_days'].mean()
                st.metric("Avg Hours (30d)", f"{avg_hours:.1f}")
            with col3:
                avg_allocation = work_data['total_allocation_percentage'].mean()
                st.metric("Avg Allocation %", f"{avg_allocation:.1f}%")
            
                
            
            # Project distribution
            fig = px.histogram(work_data, x='total_projects',
                             title="Project Count Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Allocation vs Hours scatter plot
            fig2 = px.scatter(work_data, x='total_allocation_percentage', y='hours_last_30_days',
                             hover_data=['employee_name', 'department_name'],
                             title="Allocation % vs Hours Logged (Last 30 Days)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No work profile data available.")
    
    with summary_tabs[3]:  # Attendance Report
        st.subheader("Attendance Report (Last 30 Days)")
        attend_data = get_attendance_report()
        if not attend_data.empty:
            # Add department filter
            dept_filter = st.multiselect("Filter by Department", 
                                       options=attend_data['department_name'].unique(),
                                       default=attend_data['department_name'].unique(),
                                       key="attendance_dept_filter")
            
            filtered_attend_data = attend_data[attend_data['department_name'].isin(dept_filter)]
            
            st.dataframe(filtered_attend_data, use_container_width=True)
            
            # Attendance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_attendance = filtered_attend_data['attendance_percentage'].mean()
                st.metric("Avg Attendance %", f"{avg_attendance:.1f}%")
            with col2:
                perfect_attendance = len(filtered_attend_data[filtered_attend_data['attendance_percentage'] == 100])
                st.metric("Perfect Attendance", perfect_attendance)
            with col3:
                low_attendance = len(filtered_attend_data[filtered_attend_data['attendance_percentage'] < 80])
                st.metric("Low Attendance (<80%)", low_attendance)
            with col4:
                avg_hours = filtered_attend_data['avg_hours_per_day'].mean()
                st.metric("Avg Hours/Day", f"{avg_hours:.1f}")
            
            # Attendance chart
            top_20 = filtered_attend_data.head(20)
            fig = px.bar(top_20, x='employee_name', y='attendance_percentage',
                        title="Attendance Percentage - Top 20 Employees",
                        color='attendance_percentage',
                        color_continuous_scale='RdYlGn')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Department-wise attendance
            dept_attendance = filtered_attend_data.groupby('department_name')['attendance_percentage'].mean().sort_values(ascending=False)
            
            # Create DataFrame for plotly express
            dept_attendance_df = pd.DataFrame({
                'department': dept_attendance.index,
                'avg_attendance': dept_attendance.values
            })
            
            fig2 = px.bar(dept_attendance_df, x='department', y='avg_attendance',
                         title="Average Attendance by Department",
                         labels={'department': 'Department', 'avg_attendance': 'Average Attendance (%)'})
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No attendance data available.")
    
# Export functions for use in other modules
__all__ = [
    'get_exit_report',
    'get_experience_report', 
    'get_work_profile_report',
    'get_attendance_report',
    'get_department_summary',
    'render_summary_reports'
]