import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import io
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import base64

def show_employee_master_report(engine=None, db_pool=None):
    """Display Employee Master Report with comprehensive employee and project details"""
    st.subheader("Employee Master Report")
    
    # Initialize session state
    if 'selected_employee' not in st.session_state:
        st.session_state.selected_employee = None
    if 'employee_data' not in st.session_state:
        st.session_state.employee_data = None
    if 'project_data' not in st.session_state:
        st.session_state.project_data = None
    
    # Load employee data - show all active employees by default
    employees_df = load_employee_data(engine, db_pool, "Active", "All", "All")
    
    if employees_df.empty:
        st.warning("No employees found matching the selected criteria.")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # st.subheader("Employee List")
        
        # Employee selection
        employee_options = [f"{row['employee_code']} - {row['employee_name']}" 
                          for _, row in employees_df.iterrows()]
        
        selected_employee_option = st.selectbox(
            "Select Employee:",
            employee_options,
            key="employee_selector"
        )
        
        if selected_employee_option:
            selected_employee_code = selected_employee_option.split(" - ")[0]
            
            # Load detailed employee data
            employee_details = load_employee_details(engine, db_pool, selected_employee_code)
            project_details = load_employee_projects(engine, db_pool, selected_employee_code)
            
            # Debug information
            if employee_details is None:
                st.error("Failed to load employee details. Please check the database connection.")
            else:
                st.session_state.selected_employee = selected_employee_code
                st.session_state.employee_data = employee_details
                st.session_state.project_data = project_details
    
    with col2:
        if st.session_state.employee_data is not None:
            display_employee_dashboard(st.session_state.employee_data, st.session_state.project_data)
        else:
            st.info("👈 Please select an employee from the list to view details.")
    
    # Download section
    if st.session_state.employee_data is not None:
        st.markdown("---")
        st.subheader("Download Report")
        
        col1, col2, col3 = st.columns(3)
        
        
        with col2:
            if st.button(" Download PDF Report", use_container_width=True):
                pdf_buffer = generate_pdf_report(st.session_state.employee_data, st.session_state.project_data)
                st.download_button(
                    label="Download PDF File",
                    data=pdf_buffer,
                    file_name=f"Employee_Report_{st.session_state.selected_employee}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col3:
            if st.button("Download CSV Report", use_container_width=True):
                csv_buffer = generate_csv_report(st.session_state.employee_data, st.session_state.project_data)
                st.download_button(
                    label="Download CSV File",
                    data=csv_buffer,
                    file_name=f"Employee_Report_{st.session_state.selected_employee}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def get_departments(engine, db_pool):
    """Get list of departments"""
    query = "SELECT DISTINCT department_name FROM department WHERE status = 'Active' ORDER BY department_name"
    try:
        if engine:
            df = pd.read_sql(query, engine)
        else:
            df = pd.read_sql(query, db_pool)
        return df['department_name'].tolist()
    except Exception as e:
        st.error(f"Error loading departments: {e}")
        return []

def get_business_units(engine, db_pool):
    """Get list of business units"""
    query = "SELECT DISTINCT business_unit FROM department ORDER BY business_unit"
    try:
        if engine:
            df = pd.read_sql(query, engine)
        else:
            df = pd.read_sql(query, db_pool)
        return df['business_unit'].tolist()
    except Exception as e:
        st.error(f"Error loading business units: {e}")
        return []

def load_employee_data(engine, db_pool, status_filter, dept_filter, bu_filter):
    """Load employee data with filters"""
    query = """
    SELECT 
        e.employee_code,
        e.employee_name,
        e.email,
        e.date_of_joining,
        e.employee_type,
        e.grade,
        e.status,
        d.department_name,
        d.business_unit,
        des.designation_name,
        CASE WHEN ee.employee_code IS NOT NULL THEN 'Inactive' ELSE e.status END as current_status
    FROM employee e
    LEFT JOIN department d ON e.department_id = d.department_id
    LEFT JOIN designation des ON e.designation_id = des.designation_id
    LEFT JOIN employee_exit ee ON e.employee_code = ee.employee_code
    WHERE 1=1
    """
    
    params = []
    
    if status_filter == "Active":
        query += " AND e.status = 'Active' AND ee.employee_code IS NULL"
    elif status_filter == "Inactive":
        query += " AND (e.status = 'Inactive' OR ee.employee_code IS NOT NULL)"
    
    if dept_filter != "All":
        query += " AND d.department_name = %s"
        params.append(dept_filter)
    
    if bu_filter != "All":
        query += " AND d.business_unit = %s"
        params.append(bu_filter)
    
    query += " ORDER BY e.employee_name"
    
    try:
        if engine:
            if params:
                df = pd.read_sql(query, engine, params=tuple(params))
            else:
                df = pd.read_sql(query, engine)
        else:
            if params:
                if hasattr(db_pool, 'execute'):
                    df = pd.read_sql(query, db_pool, params=tuple(params))
                else:
                    df = pd.read_sql(query, db_pool, params=params)
            else:
                df = pd.read_sql(query, db_pool)
        return df
    except Exception as e:
        # Try alternative parameter method
        try:
            if params:
                query_alt = query.replace('%s', '?')
                if engine:
                    df = pd.read_sql(query_alt, engine, params=params)
                else:
                    df = pd.read_sql(query_alt, db_pool, params=params)
            else:
                df = pd.read_sql(query, engine if engine else db_pool)
            return df
        except Exception as e2:
            st.error(f"Error loading employee data: {e}")
            st.error(f"Alternative method also failed: {e2}")
            return pd.DataFrame()

def load_employee_details(engine, db_pool, employee_code):
    """Load detailed employee information"""
    query = """
    SELECT 
        e.*,
        d.department_name,
        d.business_unit,
        des.designation_name,
        des.level as designation_level,
        ep.gender,
        ep.date_of_birth,
        ep.marital_status,
        ep.present_address,
        ep.permanent_address,
        ep.pan_number,
        ep.aadhaar_number,
        ef.bank_name,
        ef.account_number,
        ef.ifsc_code,
        mgr.employee_name as manager_name,
        ee.exit_date,
        ee.last_working_date,
        ee.exit_reason,
        ee.exit_comments
    FROM employee e
    LEFT JOIN department d ON e.department_id = d.department_id
    LEFT JOIN designation des ON e.designation_id = des.designation_id
    LEFT JOIN employee_personal ep ON e.employee_code = ep.employee_code
    LEFT JOIN employee_financial ef ON e.employee_code = ef.employee_code
    LEFT JOIN employee mgr ON e.primary_manager_id = mgr.employee_code
    LEFT JOIN employee_exit ee ON e.employee_code = ee.employee_code
    WHERE e.employee_code = %s
    """
    
    try:
        if engine:
            df = pd.read_sql(query, engine, params=(employee_code,))
        else:
            # For direct database connection, use different parameter style
            if hasattr(db_pool, 'execute'):
                # If it's a connection object
                df = pd.read_sql(query, db_pool, params=(employee_code,))
            else:
                # If it's a different type of connection, try without params keyword
                df = pd.read_sql(query, db_pool, params=[employee_code])
        return df.iloc[0] if not df.empty else None
    except Exception as e:
        # Try alternative parameter passing methods
        try:
            query_alt = query.replace('%s', '?')  # Try with ? placeholder
            if engine:
                df = pd.read_sql(query_alt, engine, params=[employee_code])
            else:
                df = pd.read_sql(query_alt, db_pool, params=[employee_code])
            return df.iloc[0] if not df.empty else None
        except Exception as e2:
            # Last resort: use string formatting (less secure but compatible)
            try:
                query_formatted = f"""
                SELECT 
                    e.*,
                    d.department_name,
                    d.business_unit,
                    des.designation_name,
                    des.level as designation_level,
                    ep.gender,
                    ep.date_of_birth,
                    ep.marital_status,
                    ep.present_address,
                    ep.permanent_address,
                    ep.pan_number,
                    ep.aadhaar_number,
                    ef.bank_name,
                    ef.account_number,
                    ef.ifsc_code,
                    mgr.employee_name as manager_name,
                    ee.exit_date,
                    ee.last_working_date,
                    ee.exit_reason,
                    ee.exit_comments
                FROM employee e
                LEFT JOIN department d ON e.department_id = d.department_id
                LEFT JOIN designation des ON e.designation_id = des.designation_id
                LEFT JOIN employee_personal ep ON e.employee_code = ep.employee_code
                LEFT JOIN employee_financial ef ON e.employee_code = ef.employee_code
                LEFT JOIN employee mgr ON e.primary_manager_id = mgr.employee_code
                LEFT JOIN employee_exit ee ON e.employee_code = ee.employee_code
                WHERE e.employee_code = '{employee_code}'
                """
                if engine:
                    df = pd.read_sql(query_formatted, engine)
                else:
                    df = pd.read_sql(query_formatted, db_pool)
                return df.iloc[0] if not df.empty else None
            except Exception as e3:
                st.error(f"Error loading employee details: {e}")
                st.error(f"Alternative method also failed: {e2}")
                st.error(f"String formatting method failed: {e3}")
                return None

def load_employee_projects(engine, db_pool, employee_code):
    """Load employee project allocation and timesheet data"""
    query = """
    SELECT 
        p.project_id,
        p.project_name,
        p.client_name,
        p.status as project_status,
        p.start_date as project_start_date,
        p.end_date as project_end_date,
        pa.allocation_percentage,
        pa.effective_from,
        pa.effective_to,
        pa.status as allocation_status,
        pa.change_reason,
        COALESCE(ts.total_hours, 0) as total_hours_logged,
        COALESCE(ts.total_days, 0) as total_days_worked,
        CASE 
            WHEN pa.effective_to IS NULL OR pa.effective_to > CURRENT_DATE 
            THEN 'Active' 
            ELSE 'Completed' 
        END as project_work_status
    FROM project_allocation pa
    JOIN project p ON pa.project_id = p.project_id
    LEFT JOIN (
        SELECT 
            project_id,
            SUM(hours_worked) as total_hours,
            COUNT(DISTINCT work_date) as total_days
        FROM timesheet 
        WHERE employee_code = %s
        GROUP BY project_id
    ) ts ON p.project_id = ts.project_id
    WHERE pa.employee_code = %s
    ORDER BY pa.effective_from DESC, p.project_name
    """
    
    try:
        if engine:
            df = pd.read_sql(query, engine, params=(employee_code, employee_code))
        else:
            # For direct database connection, use different parameter style
            if hasattr(db_pool, 'execute'):
                df = pd.read_sql(query, db_pool, params=(employee_code, employee_code))
            else:
                df = pd.read_sql(query, db_pool, params=[employee_code, employee_code])
        return df
    except Exception as e:
        # Try alternative parameter passing methods
        try:
            query_alt = query.replace('%s', '?')  # Try with ? placeholder
            if engine:
                df = pd.read_sql(query_alt, engine, params=[employee_code, employee_code])
            else:
                df = pd.read_sql(query_alt, db_pool, params=[employee_code, employee_code])
            return df
        except Exception as e2:
            # Last resort: use string formatting (less secure but compatible)
            try:
                query_formatted = f"""
                SELECT 
                    p.project_id,
                    p.project_name,
                    p.client_name,
                    p.status as project_status,
                    p.start_date as project_start_date,
                    p.end_date as project_end_date,
                    pa.allocation_percentage,
                    pa.effective_from,
                    pa.effective_to,
                    pa.status as allocation_status,
                    pa.change_reason,
                    COALESCE(ts.total_hours, 0) as total_hours_logged,
                    COALESCE(ts.total_days, 0) as total_days_worked,
                    CASE 
                        WHEN pa.effective_to IS NULL OR pa.effective_to > CURRENT_DATE 
                        THEN 'Active' 
                        ELSE 'Completed' 
                    END as project_work_status
                FROM project_allocation pa
                JOIN project p ON pa.project_id = p.project_id
                LEFT JOIN (
                    SELECT 
                        project_id,
                        SUM(hours_worked) as total_hours,
                        COUNT(DISTINCT work_date) as total_days
                    FROM timesheet 
                    WHERE employee_code = '{employee_code}'
                    GROUP BY project_id
                ) ts ON p.project_id = ts.project_id
                WHERE pa.employee_code = '{employee_code}'
                ORDER BY pa.effective_from DESC, p.project_name
                """
                if engine:
                    df = pd.read_sql(query_formatted, engine)
                else:
                    df = pd.read_sql(query_formatted, db_pool)
                return df
            except Exception as e3:
                st.error(f"Error loading project data: {e}")
                st.error(f"Alternative method also failed: {e2}")
                st.error(f"String formatting method failed: {e3}")
                return pd.DataFrame()

def display_employee_dashboard(employee_data, project_data):
    """Display comprehensive employee dashboard with proper document view"""
    
    # Convert Series to dict for easier access
    if hasattr(employee_data, 'to_dict'):
        emp_dict = employee_data.to_dict()
    else:
        emp_dict = employee_data
    
    # Helper function to safely get values
    def safe_get(key, default='N/A'):
        value = emp_dict.get(key, default)
        if pd.isna(value) or value is None:
            return default
        return str(value)
    
    # Employee Header
    st.markdown("---")
    # st.markdown("# EMPLOYEE MASTER REPORT")

    
    # Employee Basic Info Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"## 👤 {safe_get('employee_name')}")
        st.markdown(f"**Employee Code:** `{safe_get('employee_code')}`")
    with col2:
        status_color = "" if safe_get('status') == 'Active' else ""
        st.markdown(f"**Status:** {status_color} {safe_get('status')}")
    
    st.markdown("---")
    
    # SECTION 1: PERSONAL & PROFESSIONAL INFORMATION
    st.markdown("## PERSONAL & PROFESSIONAL INFORMATION")
    
    # Professional Details
    st.markdown("### Professional Details")
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    
    with prof_col1:
        st.markdown(f"""
        **Department:** {safe_get('department_name')}  
        **Business Unit:** {safe_get('business_unit')}  
        **Designation:** {safe_get('designation_name')}
        """)
    
    with prof_col2:
        st.markdown(f"""
        **Employee Type:** {safe_get('employee_type')}  
        **Grade:** {safe_get('grade')}  
        **Level:** {safe_get('level')}
        """)
    
    with prof_col3:
        total_exp = emp_dict.get('total_experience', 0)
        if pd.isna(total_exp) or total_exp is None:
            total_exp = 0
        st.markdown(f"""
        **Date of Joining:** {safe_get('date_of_joining')}  
        **Reporting Manager:** {safe_get('manager_name')}  
        **Total Experience:** {total_exp} years
        """)
    
    # Contact Information
    if safe_get('email') != 'N/A' or safe_get('mobile_number') != 'N/A':
        st.markdown("###  Contact Information")
        contact_col1, contact_col2 = st.columns(2)
        with contact_col1:
            st.markdown(f"**Email:** {safe_get('email')}")
        with contact_col2:
            st.markdown(f"**Mobile:** {safe_get('mobile_number')}")
    
    st.markdown("---")
    
    # SECTION 2: PROJECT INFORMATION
    st.markdown("## PROJECT INFORMATION")
    
    if not project_data.empty:
        # Remove duplicates based on project_name and effective dates
        project_data_clean = project_data.drop_duplicates(
            subset=['project_name', 'effective_from', 'effective_to'], 
            keep='first'
        )
        
        # Separate active and completed projects
        active_projects = project_data_clean[
            (project_data_clean['project_work_status'] == 'Active') | 
            (pd.isna(project_data_clean['effective_to']))
        ]
        completed_projects = project_data_clean[
            (project_data_clean['project_work_status'] != 'Active') & 
            (~pd.isna(project_data_clean['effective_to']))
        ]
        
        # Current/Active Projects
        st.markdown(f"### Current Projects ({len(active_projects)})")
        if not active_projects.empty:
            for idx, project in active_projects.iterrows():
                with st.container():
                    st.markdown(f"#### 📁 {project['project_name']}")
                    
                    # Project details in organized layout
                    proj_col1, proj_col2, proj_col3, proj_col4 = st.columns(4)
                    
                    with proj_col1:
                        st.markdown(f"""
                        **Client:** {project.get('client_name', 'N/A')}  
                        **Status:** Active
                        """)
                    
                    with proj_col2:
                        allocation = project.get('allocation_percentage', 0)
                        allocation_display = f"{allocation}%" if allocation else "0%"
                        st.markdown(f"""
                        **Allocation:** {allocation_display}  
                        **Duration:** {project.get('effective_from', 'N/A')} to Ongoing
                        """)
                    
                    with proj_col3:
                        hours = project.get('total_hours_logged', 0)
                        hours_display = str(hours) if hours else "0"
                        st.markdown(f"""
                        **Hours Logged:** {hours_display}  
                        **Days Worked:** {project.get('total_days_worked', 0)}
                        """)
                    
                    with proj_col4:
                        if project.get('change_reason'):
                            st.markdown(f"""
                            **Change Reason:** {project.get('change_reason', 'N/A')}
                            """)
                    
                    st.markdown("---")
        else:
            st.info("🔍 No active projects found")
        
        # Completed/Previous Projects
        st.markdown(f"### Previous Projects ({len(completed_projects)})")
        if not completed_projects.empty:
            for idx, project in completed_projects.iterrows():
                with st.container():
                    st.markdown(f"#### 📁 {project['project_name']}")
                    
                    # Project details in organized layout
                    proj_col1, proj_col2, proj_col3, proj_col4 = st.columns(4)
                    
                    with proj_col1:
                        st.markdown(f"""
                        **Client:** {project.get('client_name', 'N/A')}  
                        **Status:**  Completed
                        """)
                    
                    with proj_col2:
                        allocation = project.get('allocation_percentage', 0)
                        allocation_display = f"{allocation}%" if allocation else "0%"
                        effective_to = project.get('effective_to', 'N/A')
                        st.markdown(f"""
                        **Allocation:** {allocation_display}  
                        **Duration:** {project.get('effective_from', 'N/A')} to {effective_to}
                        """)
                    
                    with proj_col3:
                        hours = project.get('total_hours_logged', 0)
                        hours_display = str(hours) if hours else "0"
                        st.markdown(f"""
                        **Hours Logged:** {hours_display}  
                        **Days Worked:** {project.get('total_days_worked', 0)}
                        """)
                    
                    with proj_col4:
                        if project.get('change_reason'):
                            st.markdown(f"""
                            **Change Reason:** {project.get('change_reason', 'N/A')}
                            """)
                    
                    st.markdown("---")
        else:
            st.info("🔍 No previous projects found")
            
        # Project Summary
        st.markdown("### Project Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Total Projects", len(project_data_clean))
        with summary_col2:
            st.metric("Active Projects", len(active_projects))
        with summary_col3:
            st.metric("Completed Projects", len(completed_projects))
        with summary_col4:
            total_hours = project_data_clean['total_hours_logged'].fillna(0).sum()
            st.metric("Total Hours Logged", f"{total_hours}")
            
    else:
        st.info("🔍 No project information available for this employee")
    
    # SECTION 3: EXIT INFORMATION (if applicable)
    if safe_get('exit_date') != 'N/A':
        st.markdown("---")
        st.markdown("##  EXIT INFORMATION")
        
        exit_col1, exit_col2 = st.columns(2)
        with exit_col1:
            st.markdown(f"""
            **Exit Date:** {safe_get('exit_date')}  
            **Last Working Date:** {safe_get('last_working_date')}
            """)
        with exit_col2:
            st.markdown(f"""
            **Exit Reason:** {safe_get('exit_reason')}  
            **Exit Comments:** {safe_get('exit_comments')}
            """)
    
    st.markdown("---")
    st.markdown("*Report generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")

def generate_pdf_report(employee_data, project_data):
    """Generate comprehensive PDF report with professional document layout"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch,
                          leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Convert Series to dict if needed
    if hasattr(employee_data, 'to_dict'):
        emp_dict = employee_data.to_dict()
    else:
        emp_dict = employee_data
    
    # Helper function to safely get values
    def safe_get(key, default='N/A'):
        value = emp_dict.get(key, default)
        if pd.isna(value) or value is None:
            return default
        return str(value)
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceBefore=0,
        spaceAfter=24,
        fontName='Helvetica-Bold'
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.black,
        spaceBefore=20,
        spaceAfter=12,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.black,
        borderPadding=3
    )
    
    subsection_style = ParagraphStyle(
        'SubSection',
        parent=styles['Heading3'],
        fontSize=10,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold',
        leftIndent=0
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        spaceBefore=2,
        spaceAfter=2,
        leftIndent=0,
        fontName='Helvetica'
    )
    
    field_style = ParagraphStyle(
        'FieldStyle',
        parent=styles['Normal'],
        fontSize=9,
        spaceBefore=1,
        spaceAfter=1,
        leftIndent=20,
        fontName='Helvetica'
    )
    
    # Document Header
    story.append(Paragraph("EMPLOYEE MASTER REPORT", title_style))
    story.append(Paragraph("_" * 100, normal_style))
    story.append(Spacer(1, 12))
    
    # Employee Basic Information
    story.append(Paragraph(f"<b>Employee Name:</b> {safe_get('employee_name')}", normal_style))
    story.append(Paragraph(f"<b>Employee Code:</b> {safe_get('employee_code')}", normal_style))
    story.append(Paragraph(f"<b>Status:</b> {safe_get('status')}", normal_style))
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 16))
    
    # SECTION 1: PERSONAL & PROFESSIONAL INFORMATION
    story.append(Paragraph("1. PERSONAL & PROFESSIONAL INFORMATION", section_style))
    
    # Professional Details
    story.append(Paragraph("Professional Details", subsection_style))
    story.append(Paragraph(f"<b>Department:</b> {safe_get('department_name')}", field_style))
    story.append(Paragraph(f"<b>Business Unit:</b> {safe_get('business_unit')}", field_style))
    story.append(Paragraph(f"<b>Designation:</b> {safe_get('designation_name')}", field_style))
    story.append(Paragraph(f"<b>Level:</b> {safe_get('level')}", field_style))
    story.append(Paragraph(f"<b>Employee Type:</b> {safe_get('employee_type')}", field_style))
    story.append(Paragraph(f"<b>Grade:</b> {safe_get('grade')}", field_style))
    story.append(Paragraph(f"<b>Date of Joining:</b> {safe_get('date_of_joining')}", field_style))
    story.append(Paragraph(f"<b>Reporting Manager:</b> {safe_get('manager_name')}", field_style))
    story.append(Spacer(1, 8))
    
    # Experience Information
    story.append(Paragraph("Experience Details", subsection_style))
    total_exp = emp_dict.get('total_experience', 0)
    if pd.isna(total_exp) or total_exp is None:
        total_exp = 0
    
    story.append(Paragraph(f"<b>Current Experience:</b> {safe_get('current_experience')} years", field_style))
    story.append(Paragraph(f"<b>Past Experience:</b> {safe_get('past_experience')} years", field_style))
    story.append(Paragraph(f"<b>Total Experience:</b> {total_exp} years", field_style))
    story.append(Spacer(1, 8))
    
    # Contact Information
    if safe_get('email') != 'N/A' or safe_get('mobile_number') != 'N/A':
        story.append(Paragraph("Contact Information", subsection_style))
        story.append(Paragraph(f"<b>Email:</b> {safe_get('email')}", field_style))
        story.append(Paragraph(f"<b>Mobile Number:</b> {safe_get('mobile_number')}", field_style))
        story.append(Spacer(1, 8))
    
    # SECTION 2: PROJECT INFORMATION
    story.append(Paragraph("2. PROJECT INFORMATION", section_style))
    
    if not project_data.empty:
        # Remove duplicates
        project_data_clean = project_data.drop_duplicates(
            subset=['project_name', 'effective_from', 'effective_to'], 
            keep='first'
        )
        
        # Separate active and completed projects
        active_projects = project_data_clean[
            (project_data_clean['project_work_status'] == 'Active') | 
            (pd.isna(project_data_clean['effective_to']))
        ]
        completed_projects = project_data_clean[
            (project_data_clean['project_work_status'] != 'Active') & 
            (~pd.isna(project_data_clean['effective_to']))
        ]
        
        # Project Summary
        story.append(Paragraph("Project Summary", subsection_style))
        total_hours = project_data_clean['total_hours_logged'].fillna(0).sum()
        total_days = project_data_clean['total_days_worked'].fillna(0).sum()
        
        story.append(Paragraph(f"<b>Total Projects:</b> {len(project_data_clean)}", field_style))
        story.append(Paragraph(f"<b>Active Projects:</b> {len(active_projects)}", field_style))
        story.append(Paragraph(f"<b>Completed Projects:</b> {len(completed_projects)}", field_style))
        story.append(Paragraph(f"<b>Total Hours Logged:</b> {total_hours}", field_style))
        story.append(Paragraph(f"<b>Total Days Worked:</b> {total_days}", field_style))
        story.append(Spacer(1, 12))
        
        # Current Projects
        if not active_projects.empty:
            story.append(Paragraph(f"Current Projects ({len(active_projects)})", subsection_style))
            
            for idx, project in active_projects.iterrows():
                story.append(Paragraph(f"<b>Project:</b> {project['project_name']}", field_style))
                story.append(Paragraph(f"    <b>Client:</b> {project.get('client_name', 'N/A')}", field_style))
                story.append(Paragraph(f"    <b>Status:</b> Active", field_style))
                story.append(Paragraph(f"    <b>Allocation:</b> {project.get('allocation_percentage', 0)}%", field_style))
                story.append(Paragraph(f"    <b>Start Date:</b> {project.get('effective_from', 'N/A')}", field_style))
                story.append(Paragraph(f"    <b>Duration:</b> {project.get('effective_from', 'N/A')} to Ongoing", field_style))
                story.append(Paragraph(f"    <b>Hours Logged:</b> {project.get('total_hours_logged', 0)}", field_style))
                story.append(Paragraph(f"    <b>Days Worked:</b> {project.get('total_days_worked', 0)}", field_style))
                if project.get('change_reason'):
                    story.append(Paragraph(f"    <b>Change Reason:</b> {project.get('change_reason')}", field_style))
                story.append(Spacer(1, 6))
        
        # Previous Projects
        if not completed_projects.empty:
            story.append(Paragraph(f"Previous Projects ({len(completed_projects)})", subsection_style))
            
            for idx, project in completed_projects.iterrows():
                story.append(Paragraph(f"<b>Project:</b> {project['project_name']}", field_style))
                story.append(Paragraph(f"    <b>Client:</b> {project.get('client_name', 'N/A')}", field_style))
                story.append(Paragraph(f"    <b>Status:</b> Completed", field_style))
                story.append(Paragraph(f"    <b>Allocation:</b> {project.get('allocation_percentage', 0)}%", field_style))
                story.append(Paragraph(f"    <b>Duration:</b> {project.get('effective_from', 'N/A')} to {project.get('effective_to', 'N/A')}", field_style))
                story.append(Paragraph(f"    <b>Hours Logged:</b> {project.get('total_hours_logged', 0)}", field_style))
                story.append(Paragraph(f"    <b>Days Worked:</b> {project.get('total_days_worked', 0)}", field_style))
                if project.get('change_reason'):
                    story.append(Paragraph(f"    <b>Change Reason:</b> {project.get('change_reason')}", field_style))
                story.append(Spacer(1, 6))
                
    else:
        story.append(Paragraph("No project information available for this employee.", field_style))
        story.append(Spacer(1, 12))
    
    # SECTION 3: EXIT INFORMATION (if applicable)
    if safe_get('exit_date') != 'N/A':
        story.append(Paragraph("3. EXIT INFORMATION", section_style))
        story.append(Paragraph(f"<b>Exit Date:</b> {safe_get('exit_date')}", field_style))
        story.append(Paragraph(f"<b>Last Working Date:</b> {safe_get('last_working_date')}", field_style))
        story.append(Paragraph(f"<b>Exit Reason:</b> {safe_get('exit_reason')}", field_style))
        story.append(Paragraph(f"<b>Exit Comments:</b> {safe_get('exit_comments')}", field_style))
        story.append(Spacer(1, 12))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph("_" * 100, normal_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"End of Report - Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}", 
                          ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                                       alignment=TA_CENTER, fontName='Helvetica')))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_csv_report(employee_data, project_data):
    """Generate CSV report"""
    
    # Convert Series to dict if needed
    if hasattr(employee_data, 'to_dict'):
        emp_dict = employee_data.to_dict()
    else:
        emp_dict = employee_data
    
    # Helper function to safely get values
    def safe_get(key, default='N/A'):
        value = emp_dict.get(key, default)
        if pd.isna(value) or value is None:
            return default
        return str(value)
    
    # Combine employee and project data
    combined_data = []
    
    if not project_data.empty:
        for _, project in project_data.iterrows():
            row = {
                'Employee Code': safe_get('employee_code'),
                'Employee Name': safe_get('employee_name'),
                'Department': safe_get('department_name'),
                'Designation': safe_get('designation_name'),
                'Project Name': project['project_name'],
                'Client': project.get('client_name', 'N/A'),
                'Project Status': project['project_work_status'],
                'Allocation %': project.get('allocation_percentage', 0),
                'Hours Logged': project.get('total_hours_logged', 0),
                'Days Worked': project.get('total_days_worked', 0),
                'Start Date': project.get('effective_from', 'N/A'),
                'End Date': project.get('effective_to', 'N/A')
            }
            combined_data.append(row)
    else:
        # If no projects, just employee data
        combined_data.append({
            'Employee Code': safe_get('employee_code'),
            'Employee Name': safe_get('employee_name'),
            'Department': safe_get('department_name'),
            'Designation': safe_get('designation_name'),
            'Project Name': 'No Projects',
            'Client': 'N/A',
            'Project Status': 'N/A',
            'Allocation %': 0,
            'Hours Logged': 0,
            'Days Worked': 0,
            'Start Date': 'N/A',
            'End Date': 'N/A'
        })
    
    df = pd.DataFrame(combined_data)
    return df.to_csv(index=False).encode('utf-8')