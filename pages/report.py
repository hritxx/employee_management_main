import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from pages.employee_master import show_employee_master_report
import plotly.express as px
import plotly.graph_objects as go

def create_project_document_report(project_data, project_info, weekly_hours_data, title, engine=None, db_pool=None):
    """Create a comprehensive document-style PDF report with clean, non-repetitive content"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        textColor=colors.HexColor('#1f4e79'),
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=12,
        textColor=colors.HexColor('#2c5aa0')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=8,
        textColor=colors.HexColor('#4472c4')
    )
    
    # Title
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 12))
    
    # Date and basic info
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Project Overview Section
    if not project_info.empty:
        elements.append(Paragraph("PROJECT OVERVIEW", heading_style))
        proj_info = project_info.iloc[0]
        # Fetch manager using primary_manager_id
        manager = get_project_manager(proj_info.get('project_id'), engine, db_pool)
        manager_name = manager['employee_name'] if manager else 'N/A'
        # Total hours for project (match UI col4)
        total_hours = None
        if not weekly_hours_data.empty:
            total_hours = weekly_hours_data['hours_worked'].sum()
        elif not project_data.empty:
            try:
                project_id = proj_info.get('project_id', None)
                if project_id:
                    total_hours_query = f"""
                    SELECT ROUND(SUM(t.hours_worked)::numeric, 2) as total_hours
                    FROM timesheet t
                    WHERE t.project_id = '{project_id}'
                    """
                    total_hours_result = run_query(total_hours_query, engine, db_pool)
                    if not total_hours_result.empty and total_hours_result.iloc[0, 0] is not None:
                        total_hours = float(total_hours_result.iloc[0, 0])
            except Exception:
                total_hours = 0
        if total_hours is None:
            total_hours = 0
        # Team counts
        total_team_members = project_data['employee_code'].nunique()
        current_team_members = project_data[(project_data['allocation_status'] == 'Active') & (project_data['effective_to'].isna())]['employee_code'].nunique()
        overview_text = f"""
        <b>Project Name:</b> {proj_info.get('project_name', 'N/A')}<br/>
        <b>Project ID:</b> {proj_info.get('project_id', 'N/A')}<br/>
        <b>Client:</b> {proj_info.get('client_name', 'N/A')}<br/>
        <b>Status:</b> {proj_info.get('status', 'N/A')}<br/>
        <b>Start Date:</b> {proj_info.get('start_date', 'N/A')}<br/>
        <b>End Date:</b> {proj_info.get('end_date', 'N/A') if proj_info.get('end_date') else 'Ongoing'}<br/>
        <b>Project Manager:</b> {manager_name}<br/>
        <b>Total Hours:</b> {total_hours:.1f}<br/>
        <b>Total Team Members:</b> {total_team_members}<br/>
        <b>Current Team Members:</b> {current_team_members}<br/>
        """
        elements.append(Paragraph(overview_text, styles['Normal']))
        elements.append(Spacer(1, 16))
    
    # Project Manager Section
    if not project_data.empty:
        managers = project_data[project_data['employee_type'].str.contains('Manager|Lead', case=False, na=False)]
        if not managers.empty:
            elements.append(Paragraph("PROJECT MANAGEMENT", heading_style))
            # Get unique manager (in case of duplicates)
            manager = managers.drop_duplicates('employee_code').iloc[0]
            manager_text = f"""
            <b>Project Manager:</b> {manager['employee_name']} ({manager['employee_code']})<br/>
            <b>Department:</b> {manager.get('department_name', 'N/A')}<br/>
            <b>Designation:</b> {manager.get('designation_name', 'N/A')}<br/>
            """
            elements.append(Paragraph(manager_text, styles['Normal']))
            elements.append(Spacer(1, 16))
    
    # Team Composition Section
    if not project_data.empty:
        elements.append(Paragraph("TEAM COMPOSITION & ALLOCATION HISTORY", heading_style))

        # Separate current and past employees
        current_employees = project_data[
            (project_data['allocation_status'] == 'Active') & 
            (project_data['effective_to'].isna())
        ]['employee_code'].unique()
        past_employees = project_data[
            ~project_data['employee_code'].isin(current_employees)
        ]['employee_code'].unique()

        # Helper to get total hours and task summaries for an employee
        def get_emp_stats(emp_code, engine, db_pool):
            # Get total hours for this employee in this project from timesheet
            project_id = project_info.iloc[0]['project_id'] if not project_info.empty else None
            total_hours = 0
            if project_id:
                try:
                    emp_hours_query = f"""
                    SELECT SUM(hours_worked) as total_hours
                    FROM timesheet
                    WHERE employee_code = '{emp_code}' AND project_id = '{project_id}'
                    """
                    emp_hours_result = run_query(emp_hours_query, engine, db_pool)
                    if not emp_hours_result.empty and emp_hours_result.iloc[0, 0] is not None:
                        total_hours = float(emp_hours_result.iloc[0, 0])
                except Exception:
                    total_hours = 0
            # Fetch saved summary from task_summary table
            saved_summary = get_employee_task_summary(emp_code, project_id, engine, db_pool) if project_id else None
            if saved_summary:
                task_summaries = [saved_summary]
            else:
                # Fallback to unique task descriptions from timesheet
                emp_hours = weekly_hours_data[weekly_hours_data['employee_code'] == emp_code]
                task_summaries = emp_hours['task_description'].dropna().unique()
            return total_hours, task_summaries

        # --- Current Employees ---
        if len(current_employees) > 0:
            elements.append(Paragraph("<b>Current Team Members</b>", styles['Normal']))
            for emp_code in current_employees:
                emp_data = project_data[project_data['employee_code'] == emp_code]
                emp_info = emp_data.iloc[0]
                total_hours, task_summaries = get_emp_stats(emp_code, engine, db_pool)
                elements.append(Paragraph(f"{emp_info['employee_name']} ({emp_code})", subheading_style))
                emp_details = f"""
                <b>Designation:</b> {emp_info.get('designation_name', 'N/A')}<br/>
                <b>Department:</b> {emp_info.get('department_name', 'N/A')}<br/>
                <b>Employee Type:</b> {emp_info.get('employee_type', 'N/A')}<br/>
                <b>Total Experience:</b> {emp_info.get('total_experience', 'N/A')} years<br/>
                <b>Total Hours:</b> {total_hours:.1f}<br/>
                <b>Task Summaries:</b> {', '.join(task_summaries) if len(task_summaries) else 'N/A'}<br/>
                """
                elements.append(Paragraph(emp_details, styles['Normal']))
                elements.append(Spacer(1, 8))
                allocation_columns = ['effective_from', 'effective_to', 'allocation_percentage', 'allocation_status', 'change_reason']
                unique_allocations = emp_data[allocation_columns].drop_duplicates()
                if len(unique_allocations) > 0:
                    elements.append(Paragraph("<b>Allocation History:</b>", styles['Normal']))
                    for _, allocation in unique_allocations.iterrows():
                        duration_text = f"From {allocation['effective_from']}"
                        if pd.notna(allocation['effective_to']) and allocation['effective_to']:
                            duration_text += f" to {allocation['effective_to']}"
                        else:
                            duration_text += " (Current)"
                        alloc_text = f"""
                        • <b>Period:</b> {duration_text}<br/>
                          <b>Allocation:</b> {allocation['allocation_percentage']}%<br/>
                          <b>Status:</b> {allocation['allocation_status']}<br/>
                        """
                        if pd.notna(allocation.get('change_reason')) and allocation.get('change_reason'):
                            alloc_text += f"      <b>Reason:</b> {allocation['change_reason']}<br/>"
                        elements.append(Paragraph(alloc_text, styles['Normal']))
                elements.append(Spacer(1, 12))

        # --- Past Employees ---
        if len(past_employees) > 0:
            elements.append(Paragraph("<b>Past Team Members</b>", styles['Normal']))
            for emp_code in past_employees:
                emp_data = project_data[project_data['employee_code'] == emp_code]
                emp_info = emp_data.iloc[0]
                total_hours, task_summaries = get_emp_stats(emp_code, engine, db_pool)
                elements.append(Paragraph(f"{emp_info['employee_name']} ({emp_code})", subheading_style))
                emp_details = f"""
                <b>Designation:</b> {emp_info.get('designation_name', 'N/A')}<br/>
                <b>Department:</b> {emp_info.get('department_name', 'N/A')}<br/>
                <b>Employee Type:</b> {emp_info.get('employee_type', 'N/A')}<br/>
                <b>Total Experience:</b> {emp_info.get('total_experience', 'N/A')} years<br/>
                <b>Total Hours:</b> {total_hours:.1f}<br/>
                <b>Task Summaries:</b> {', '.join(task_summaries) if len(task_summaries) else 'N/A'}<br/>
                """
                elements.append(Paragraph(emp_details, styles['Normal']))
                elements.append(Spacer(1, 8))
                allocation_columns = ['effective_from', 'effective_to', 'allocation_percentage', 'allocation_status', 'change_reason']
                unique_allocations = emp_data[allocation_columns].drop_duplicates()
                if len(unique_allocations) > 0:
                    elements.append(Paragraph("<b>Allocation History:</b>", styles['Normal']))
                    for _, allocation in unique_allocations.iterrows():
                        duration_text = f"From {allocation['effective_from']}"
                        if pd.notna(allocation['effective_to']) and allocation['effective_to']:
                            duration_text += f" to {allocation['effective_to']}"
                        else:
                            duration_text += " (Current)"
                        alloc_text = f"""
                        • <b>Period:</b> {duration_text}<br/>
                          <b>Allocation:</b> {allocation['allocation_percentage']}%<br/>
                          <b>Status:</b> {allocation['allocation_status']}<br/>
                        """
                        if pd.notna(allocation.get('change_reason')) and allocation.get('change_reason'):
                            alloc_text += f"      <b>Reason:</b> {allocation['change_reason']}<br/>"
                        elements.append(Paragraph(alloc_text, styles['Normal']))
                elements.append(Spacer(1, 12))
    
    # Weekly Hours Analysis Section
    if not weekly_hours_data.empty:
        elements.append(Paragraph("WEEKLY HOURS ANALYSIS", heading_style))
        
        # Get unique employees from hours data
        unique_hour_employees = weekly_hours_data['employee_code'].unique()
        
        for emp_code in unique_hour_employees:
            emp_hours = weekly_hours_data[weekly_hours_data['employee_code'] == emp_code]
            emp_name = emp_hours.iloc[0]['employee_name']
            
            elements.append(Paragraph(f"{emp_name} ({emp_code}) - Weekly Hours Breakdown", subheading_style))
            
            # Convert to datetime for proper grouping
            emp_hours = emp_hours.copy()
            emp_hours['work_date'] = pd.to_datetime(emp_hours['work_date'])
            
            # Group by week (Monday as week start)
            emp_hours['week_start'] = emp_hours['work_date'] - pd.to_timedelta(emp_hours['work_date'].dt.dayofweek, unit='d')
            
            weekly_summary = emp_hours.groupby('week_start').agg({
                'hours_worked': 'sum',
                'work_date': 'count'
            }).rename(columns={'work_date': 'days_worked'}).sort_index()
            
            # Display weekly summary
            for week_start, week_data in weekly_summary.iterrows():
                week_end = week_start + timedelta(days=6)
                avg_hours_per_day = week_data['hours_worked'] / week_data['days_worked'] if week_data['days_worked'] > 0 else 0
                
                week_text = f"""
                <b>Week of {week_start.strftime('%B %d, %Y')}:</b><br/>
                • Total Hours: {week_data['hours_worked']:.1f}<br/>
                • Days Worked: {week_data['days_worked']}<br/>
                • Average Hours/Day: {avg_hours_per_day:.1f}<br/>
                """
                elements.append(Paragraph(week_text, styles['Normal']))
            
            # Monthly summary for this employee
            emp_hours['month'] = emp_hours['work_date'].dt.to_period('M')
            monthly_summary = emp_hours.groupby('month')['hours_worked'].sum().sort_index()
            
            if not monthly_summary.empty:
                elements.append(Paragraph("<b>Monthly Summary:</b>", styles['Normal']))
                for month, total_hours in monthly_summary.items():
                    month_text = f"• {month.strftime('%B %Y')}: {total_hours:.1f} hours"
                    elements.append(Paragraph(month_text, styles['Normal']))
            
            elements.append(Spacer(1, 16))
    
    
    # Summary Section
    elements.append(Paragraph("SUMMARY", heading_style))
    summary_text = """
    This report provides a comprehensive overview of the project including team composition, 
    allocation history, and work hours analysis. All data is presented chronologically and 
    represents the current state of project assignments and historical changes.
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    
    # Build and return the document
    doc.build(elements)
    buffer.seek(0)
    return buffer
def render_standard_reports(engine=None, db_pool=None):
    """Main function to display standard reports - called from app.py tab 5"""
    st.header("Standard Reports")
    
    # Create sub-tabs for different report types
    report_tabs = st.tabs(["📁 Project Master", "👥 Employee Master"])
    
    # Tab 1: Project Master Report
    with report_tabs[0]:
        show_project_master_report(engine, db_pool)
    
    # Tab 2: Employee Master Report
    with report_tabs[1]:
        show_employee_master_report(engine, db_pool)

def show_project_master_report(engine=None, db_pool=None):
    """Display Enhanced Project Master Report"""
    st.subheader("Project Master Report")

    
    # Get projects from database
    try:
        projects_df = run_query("SELECT project_id, project_name, client_name, status FROM project ORDER BY project_name", engine, db_pool)
        
        if not projects_df.empty:
            project_options = ["Select a Project"] + [f"{row['project_name']} ({row['project_id']})" for _, row in projects_df.iterrows()]
            selected_project = st.selectbox("Choose Project", project_options, key="project_master_selector")
            
            if selected_project != "Select a Project":
                # Extract project_id from selection
                project_id = selected_project.split("(")[-1].rstrip(")")
                project_name = selected_project.split(" (")[0]
                
                # Remove date range filter UI; use project start/end dates from project_info
                project_info = get_project_info(project_id, engine, db_pool)
                if not project_info.empty:
                    proj_info = project_info.iloc[0]
                    # Use project start and end dates for data queries
                    start_date = proj_info.get('start_date', None)
                    end_date = proj_info.get('end_date', None)
                    if not start_date:
                        start_date = datetime.now() - timedelta(days=90)
                    if not end_date or pd.isna(end_date):
                        end_date = datetime.now().date()
                else:
                    # Fallback if project_info is empty
                    start_date = datetime.now() - timedelta(days=90)
                    end_date = datetime.now().date()
                # Get comprehensive project data
                project_data = get_project_allocation_history(project_id, engine, db_pool)
                weekly_hours_data = get_project_weekly_hours(project_id, start_date, end_date, engine, db_pool)
                
                if not project_data.empty:
                    # Display project overview
                    st.markdown("<h5 style='margin-bottom:0.5em;'>Project Overview</h5>", unsafe_allow_html=True)
                    # Project Overview - Professional 3-column layout
                    if not project_info.empty:
                        proj_info = project_info.iloc[0]
                        manager = get_project_manager(proj_info.get('project_id'), engine, db_pool)
                        manager_name = manager['employee_name'] if manager else 'N/A'
                        # Calculate team stats
                        total_members = project_data['employee_code'].nunique()
                        current_date = pd.Timestamp.now().date()
                        current_members = project_data[
                            (project_data['effective_from'] <= current_date) &
                            ((project_data['effective_to'].isna()) | (project_data['effective_to'] >= current_date)) &
                            (project_data['allocation_status'] == 'Active')
                        ]['employee_code'].nunique()
                        # Total hours
                        total_hours = 0
                        try:
                            total_hours_query = f"""
                            SELECT ROUND(SUM(t.hours_worked)::numeric, 2) as total_hours
                            FROM timesheet t
                            WHERE t.project_id = '{project_id}'
                            """
                            total_hours_result = run_query(total_hours_query, engine, db_pool)
                            if not total_hours_result.empty and total_hours_result.iloc[0, 0] is not None:
                                total_hours = total_hours_result.iloc[0, 0]
                                if pd.isna(total_hours):
                                    total_hours = 0
                        except Exception:
                            total_hours = 0
                        # 3-column layout
                        prof_col1, prof_col2, prof_col3 = st.columns(3)
                        with prof_col1:
                            st.markdown(f"""
                            **Project Name:** {proj_info.get('project_name', 'N/A')}  
                            **Project ID:** {proj_info.get('project_id', 'N/A')}  
                            **Client:** {proj_info.get('client_name', 'N/A')}  
                            **Status:** {proj_info.get('status', 'N/A')}
                            """)
                        with prof_col2:
                            st.markdown(f"""
                            **Start Date:** {proj_info.get('start_date', 'N/A')}  
                            **End Date:** {proj_info.get('end_date', 'Ongoing') if proj_info.get('end_date') else 'Ongoing'}  
                            **Project Manager:** {manager_name}
                            """)
                        with prof_col3:
                            st.markdown(f"""
                            **Total Hours:** {float(total_hours):.1f}  
                            **Total Team Members:** {total_members}  
                            **Current Team Members:** {current_members}
                            """)

                    # Team composition with segregation
                    st.markdown("###  Team Composition & History")
                    
                    # Separate current and past employees
                    current_employees = project_data[
                        (project_data['allocation_status'] == 'Active') & 
                        (project_data['effective_to'].isna())
                    ]['employee_code'].unique()
                    
                    past_employees = project_data[
                        ~project_data['employee_code'].isin(current_employees)
                    ]['employee_code'].unique()
                    
                    # Remove duplicates from project_data
                    project_data_clean = project_data.drop_duplicates(
                        subset=['employee_code', 'effective_from', 'effective_to', 'allocation_percentage']
                    ).sort_values(['employee_code', 'effective_from'])
                    
                    # Current Team Members
                    if len(current_employees) > 0:
                        st.markdown("####  **Current Team Members**")
                        for emp_code in current_employees:
                            emp_data = project_data_clean[project_data_clean['employee_code'] == emp_code]
                            emp_info = emp_data.iloc[0]
                            # Get current allocation
                            current_allocation = emp_data[
                                (emp_data['allocation_status'] == 'Active') & 
                                (emp_data['effective_to'].isna())
                            ]
                            current_alloc_pct = current_allocation['allocation_percentage'].iloc[0] if not current_allocation.empty else 0
                            # Calculate total hours and task summaries
                            # --- FIX: Always fetch total hours from timesheet for this employee/project ---
                            total_hours = 0
                            try:
                                emp_hours_query = f"""
                                SELECT SUM(hours_worked) as total_hours
                                FROM timesheet
                                WHERE employee_code = '{emp_code}' AND project_id = '{project_id}'
                                """
                                emp_hours_result = run_query(emp_hours_query, engine, db_pool)
                                if not emp_hours_result.empty and emp_hours_result.iloc[0, 0] is not None:
                                    total_hours = float(emp_hours_result.iloc[0, 0])
                            except Exception:
                                total_hours = 0
                            # Fetch saved summary from task_summary table for UI
                            saved_summary = get_employee_task_summary(emp_code, project_id, engine, db_pool) if 'get_employee_task_summary' in globals() else None
                            if saved_summary:
                                task_summaries = [saved_summary]
                            else:
                                emp_hours = weekly_hours_data[weekly_hours_data['employee_code'] == emp_code]
                                task_summaries = emp_hours['task_description'].dropna().unique()
                            with st.expander(f"👤 {emp_info['employee_name']} ({emp_code}) - **{current_alloc_pct}% allocated**"):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"""
                                    **Personal Details:**
                                    - **Designation:** {emp_info.get('designation_name', 'N/A')}
                                    - **Department:** {emp_info.get('department_name', 'N/A')}
                                    - **Employee Type:** {emp_info.get('employee_type', 'N/A')}
                                    - **Total Experience:** {emp_info.get('total_experience', 'N/A')} years
                                    - **Total Hours:** {total_hours:.1f}
                                    - **Task Summaries:** {', '.join(task_summaries) if len(task_summaries) else 'N/A'}

                                    - **Allocation History:**
                                    """)
                                    for _, allocation in emp_data.iterrows():
                                        duration_text = f"From {allocation['effective_from']}"
                                        if pd.notna(allocation['effective_to']):
                                            duration_text += f" to {allocation['effective_to']}"
                                            days = (pd.to_datetime(allocation['effective_to']) - pd.to_datetime(allocation['effective_from'])).days
                                            duration_text += f" ({days} days)"
                                        else:
                                            duration_text += " (Current)"
                                            days = (datetime.now().date() - pd.to_datetime(allocation['effective_from']).date()).days
                                            duration_text += f" ({days} days so far)"
                                        st.markdown(f"""
                                        **Period:** {duration_text}  
                                        &nbsp;&nbsp;&nbsp;&nbsp;**Allocation:** {allocation['allocation_percentage']}%  
                                        &nbsp;&nbsp;&nbsp;&nbsp;**Status:** {allocation['allocation_status']}""")
                                        if allocation.get('change_reason'):
                                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Reason:** {allocation['change_reason']}")
                                        st.markdown("")
                                with col2:
                                    # Weekly hours for this employee
                                    emp_hours = weekly_hours_data[weekly_hours_data['employee_code'] == emp_code]
                                    if not emp_hours.empty:
                                        st.markdown("**Recent Hours Summary:**")
                                        emp_hours['week'] = pd.to_datetime(emp_hours['work_date']).dt.isocalendar().week
                                        emp_hours['year'] = pd.to_datetime(emp_hours['work_date']).dt.year
                                        weekly_summary = emp_hours.groupby(['year', 'week']).agg({
                                            'hours_worked': 'sum',
                                            'work_date': 'count'
                                        }).rename(columns={'work_date': 'days_worked'}).tail(4)
                                        st.metric("Total Hours", f"{total_hours:.1f}h")
                                        for (year, week), week_data in weekly_summary.iterrows():
                                            st.markdown(f"""
                                            **Week {week}, {year}:**  
                                            Hours: {week_data['hours_worked']:.1f}h | Days: {week_data['days_worked']} | Avg: {week_data['hours_worked']/week_data['days_worked']:.1f}h/day
                                            """)
                                    else:
                                        st.markdown(" ")
                    # Past Team Members
                    if len(past_employees) > 0:
                        st.markdown("####  **Past Team Members**")
                        for emp_code in past_employees:
                            emp_data = project_data_clean[project_data_clean['employee_code'] == emp_code]
                            emp_info = emp_data.iloc[0]
                            # Calculate total days worked
                            total_days = 0
                            for _, allocation in emp_data.iterrows():
                                if pd.notna(allocation['effective_to']):
                                    effective_to = pd.to_datetime(allocation['effective_to'])
                                    effective_from = pd.to_datetime(allocation['effective_from'])
                                    days = (effective_to - effective_from).days
                                    total_days += days
                            # Calculate total hours and task summaries
                            # --- FIX: Always fetch total hours from timesheet for this employee/project ---
                            total_hours = 0
                            try:
                                emp_hours_query = f"""
                                SELECT SUM(hours_worked) as total_hours
                                FROM timesheet
                                WHERE employee_code = '{emp_code}' AND project_id = '{project_id}'
                                """
                                emp_hours_result = run_query(emp_hours_query, engine, db_pool)
                                if not emp_hours_result.empty and emp_hours_result.iloc[0, 0] is not None:
                                    total_hours = float(emp_hours_result.iloc[0, 0])
                            except Exception:
                                total_hours = 0
                            saved_summary = get_employee_task_summary(emp_code, project_id, engine, db_pool) if 'get_employee_task_summary' in globals() else None
                            if saved_summary:
                                task_summaries = [saved_summary]
                            else:
                                emp_hours = weekly_hours_data[weekly_hours_data['employee_code'] == emp_code]
                                task_summaries = emp_hours['task_description'].dropna().unique()
                            with st.expander(f"👤 {emp_info['employee_name']} ({emp_code}) - **Worked {total_days} days**"):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"""
                                    **Personal Details:**
                                    - **Designation:** {emp_info.get('designation_name', 'N/A')}
                                    - **Department:** {emp_info.get('department_name', 'N/A')}
                                    - **Employee Type:** {emp_info.get('employee_type', 'N/A')}
                                    - **Total Experience:** {emp_info.get('total_experience', 'N/A')} years
                                    - **Total Hours:** {total_hours:.1f}
                                    - **Task Summaries:** {', '.join(task_summaries) if len(task_summaries) else 'N/A'}
                                     Historical Allocation:
                                    """)
                                    for _, allocation in emp_data.iterrows():
                                        if pd.notna(allocation['effective_to']):
                                            effective_to = pd.to_datetime(allocation['effective_to'])
                                            effective_from = pd.to_datetime(allocation['effective_from'])
                                            duration_text = f"From {allocation['effective_from']} to {allocation['effective_to']}"
                                            days = (effective_to - effective_from).days
                                            duration_text += f" ({days} days)"
                                            st.markdown(f"""
                                            **Period:** {duration_text}  
                                            &nbsp;&nbsp;&nbsp;&nbsp;**Allocation:** {allocation['allocation_percentage']}%  
                                            &nbsp;&nbsp;&nbsp;&nbsp;**Status:** {allocation['allocation_status']}""")
                                            if allocation.get('change_reason'):
                                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Reason:** {allocation['change_reason']}")
                                            st.markdown("")
                                with col2:
                                    emp_hours = weekly_hours_data[weekly_hours_data['employee_code'] == emp_code]
                                    if not emp_hours.empty:
                                        st.markdown("**Historical Hours:**")
                                        st.metric("Total Hours", f"{total_hours:.1f}h")
                                        avg_hours_per_day = emp_hours['hours_worked'].mean()
                                        st.metric("Avg Hours/Day", f"{avg_hours_per_day:.1f}h")
                                    else:
                                        st.markdown(" ")
                    # Hours analysis chart
                    if not weekly_hours_data.empty:
                        st.markdown("###  Hours Analysis")
                        
                        # Weekly hours chart
                        weekly_chart_data = weekly_hours_data.copy()
                        weekly_chart_data['week_start'] = pd.to_datetime(weekly_chart_data['work_date']) - pd.to_timedelta(pd.to_datetime(weekly_chart_data['work_date']).dt.dayofweek, unit='d')
                        weekly_totals = weekly_chart_data.groupby(['week_start', 'employee_name'])['hours_worked'].sum().reset_index()
                        
                        fig = px.bar(weekly_totals, x='week_start', y='hours_worked', color='employee_name',
                                   title='Weekly Hours by Employee', labels={'week_start': 'Week', 'hours_worked': 'Hours'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download options
                    st.markdown("### Download Report")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(" Generate Document Report", key="gen_proj_doc_report"):
                            try:
                                pdf_buffer = create_project_document_report(
                                    project_data, project_info, weekly_hours_data,
                                    f"Project Master Report - {project_name}",
                                    engine, db_pool
                                )
                                st.download_button(
                                    label=" Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=f"project_report_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    key="download_proj_doc_pdf"
                                )
                                
                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")
                    
                    with col2:
                        # CSV export for raw data
                        combined_data = project_data.merge(
                            weekly_hours_data.groupby('employee_code').agg({
                                'hours_worked': 'sum'
                            }).reset_index(),
                            on='employee_code', how='left'
                        )
                        
                        csv = combined_data.to_csv(index=False)
                        st.download_button(
                            label="Download Raw Data (CSV)",
                            data=csv,
                            file_name=f"project_data_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_proj_raw_csv"
                        )
                
                else:
                    st.warning("⚠️ No allocation data found for this project.")
        else:
            st.warning("⚠️ No projects found in the database.")
            
    except Exception as e:
        st.error(f" Error loading projects: {str(e)}")



# Enhanced database functions
def run_query(query, engine=None, db_pool=None):
    """Execute SQL query and return results as DataFrame"""
    try:
        if engine is not None:
            df = pd.read_sql(query, engine)
        elif db_pool is not None:
            # Check for valid connection/engine type
            if hasattr(db_pool, 'execute') or hasattr(db_pool, 'connect'):
                df = pd.read_sql(query, db_pool)
            else:
                st.error(f"db_pool is not a valid SQLAlchemy connection or engine. Type: {type(db_pool)}. Value: {db_pool}")
                return pd.DataFrame()
        else:
            # Return empty DataFrame if no connection
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Database query error: {str(e)}")
        return pd.DataFrame()

def get_project_info(project_id, engine=None, db_pool=None):
    """Get basic project information"""
    query = f"""
    SELECT project_id, project_name, client_name, status, start_date, end_date
    FROM project 
    WHERE project_id = '{project_id}'
    """
    return run_query(query, engine, db_pool)

def get_project_allocation_history(project_id, engine=None, db_pool=None):
    """Get comprehensive project allocation history with employee details"""
    query = f"""
    SELECT 
        pa.allocation_id,
        pa.employee_code,
        e.employee_name,
        e.employee_type,
        e.total_experience,
        d.department_name,
        des.designation_name,
        pa.project_id,
        p.project_name,
        pa.allocation_percentage,
        pa.effective_from,
        pa.effective_to,
        pa.status as allocation_status,
        pa.change_reason,
        pa.created_at
    FROM project_allocation pa
    JOIN employee e ON pa.employee_code = e.employee_code
    JOIN project p ON pa.project_id = p.project_id
    LEFT JOIN department d ON e.department_id = d.department_id
    LEFT JOIN designation des ON e.designation_id = des.designation_id
    WHERE pa.project_id = '{project_id}'
    ORDER BY e.employee_name, pa.effective_from DESC
    """
    return run_query(query, engine, db_pool)

def get_project_weekly_hours(project_id, start_date, end_date, engine=None, db_pool=None):
    """Get weekly hours data for project within date range"""
    query = f"""
    SELECT 
        t.employee_code,
        e.employee_name,
        t.work_date,
        t.hours_worked,
        t.task_description,
        t.project_id
    FROM timesheet t
    JOIN employee e ON t.employee_code = e.employee_code
    WHERE t.project_id = '{project_id}'
    AND t.work_date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY t.employee_code, t.work_date
    """
    return run_query(query, engine, db_pool)

def get_project_hours_by_employee(project_id, engine=None, db_pool=None):
    """Get hours breakdown by employee for a specific project"""
    query = f"""
    SELECT
        e.employee_name,
        ROUND(SUM(t.hours_worked)::numeric, 2) AS total_hours
    FROM timesheet AS t
    JOIN employee AS e
        ON t.employee_code = e.employee_code
    WHERE
        t.project_id = '{project_id}'
    GROUP BY
        e.employee_name
    ORDER BY
        total_hours DESC
    """
    return run_query(query, engine, db_pool)

def get_employee_task_summary(employee_code, project_id, engine=None, db_pool=None):
    """Fetch the saved task summary for an employee on a project from the task_summary table."""
    query = f"""
    SELECT summary_text
    FROM task_summary
    WHERE employee_code = '{employee_code}' AND project_id = '{project_id}' AND status = 'Active'
    ORDER BY generated_at DESC
    LIMIT 1
    """
    try:
        df = run_query(query, engine, db_pool)
        if not df.empty and 'summary_text' in df.columns:
            return df.iloc[0]['summary_text']
    except Exception as e:
        st.error(f"Error fetching task summary: {str(e)}")
    return None

def get_project_manager(project_id, engine=None, db_pool=None):
    """Fetch project manager from the project table's manager_id field."""
    query = f"""
    SELECT e.employee_code, e.employee_name, d.department_name, des.designation_name
    FROM project p
    JOIN employee e ON p.manager_id = e.employee_code
    LEFT JOIN department d ON e.department_id = d.department_id
    LEFT JOIN designation des ON e.designation_id = des.designation_id
    WHERE p.project_id = '{project_id}'
    """
    df = run_query(query, engine, db_pool)
    if not df.empty:
        return df.iloc[0].to_dict()
    
    # Fallback to the old method if manager_id is not set
    query = f"""
    SELECT e.employee_code, e.employee_name, d.department_name, des.designation_name
    FROM project_allocation pa
    JOIN employee e ON pa.employee_code = e.employee_code
    LEFT JOIN department d ON e.department_id = d.department_id
    LEFT JOIN designation des ON e.designation_id = des.designation_id
    WHERE pa.project_id = '{project_id}'
      AND (LOWER(e.employee_type) LIKE '%manager%' OR LOWER(e.employee_type) LIKE '%lead%')
    ORDER BY pa.effective_from DESC
    LIMIT 1
    """
    df = run_query(query, engine, db_pool)
    if not df.empty:
        return df.iloc[0].to_dict()
    return None
