import streamlit as st
import psycopg2
import pandas as pd
import google.generativeai as genai
from typing import List, Dict, Optional
import json
import os
from datetime import datetime, date
import re
from datetime import date, datetime

class TaskSummarizer:
    def __init__(self):
        """Initialize the task summarizer with environment variables"""
        # Get API key from environment
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            st.error("GEMINI_API_KEY not found in environment variables")
            return
            
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Database configuration from environment variables
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'hr_admin'),
            'password': os.getenv('DB_PASSWORD'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
    def get_database_connection(self):
        """Create database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return None
    
    def fetch_employee_timesheet_data(self, employee_code: str) -> Dict:
        """Fetch all timesheet data for a given employee grouped by project"""
        conn = self.get_database_connection()
        if not conn:
            return {}
        
        try:
            query = """
            SELECT 
                t.project_id,
                p.project_name,
                p.client_name,
                t.task_description,
                t.hours_worked,
                t.work_date,
                SUM(t.hours_worked) OVER (PARTITION BY t.project_id) as total_project_hours,
                MIN(t.work_date) OVER (PARTITION BY t.project_id) as project_start_date,
                MAX(t.work_date) OVER (PARTITION BY t.project_id) as project_end_date
            FROM timesheet t
            LEFT JOIN project p ON t.project_id = p.project_id
            WHERE t.employee_code = %s 
                AND t.task_description IS NOT NULL 
                AND t.task_description != ''
            ORDER BY t.project_id, t.work_date DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[employee_code])
            conn.close()
            
            if df.empty:
                return {}
            
            # Group by project
            projects_data = {}
            for project_id, group in df.groupby('project_id'):
                project_info = {
                    'project_name': group.iloc[0]['project_name'],
                    'client_name': group.iloc[0]['client_name'],
                    'total_hours': group.iloc[0]['total_project_hours'],
                    'start_date': group.iloc[0]['project_start_date'],
                    'end_date': group.iloc[0]['project_end_date'],
                    'tasks': []
                }
                
                for _, row in group.iterrows():
                    project_info['tasks'].append({
                        'description': row['task_description'],
                        'hours': row['hours_worked'],
                        'date': row['work_date']
                    })
                
                projects_data[project_id] = project_info
            
            return projects_data
        
        except Exception as e:
            st.error(f"Error fetching timesheet data: {str(e)}")
            if conn:
                conn.close()
            return {}
    
    def summarize_project_tasks_with_gemini(self, project_data: Dict) -> str:
        """Summarize tasks for a specific project using Gemini API"""
        if not project_data or not project_data.get('tasks'):
            return "No tasks found for this project."
        
        # Prepare tasks text
        tasks_text = []
        for task in project_data['tasks']:
            tasks_text.append(f"- {task['description']} ({task['hours']} hours on {task['date']})")
        
        all_tasks = "\n".join(tasks_text)
        total_hours = project_data['total_hours']
        project_name = project_data.get('project_name', 'Unknown Project')
        client_name = project_data.get('client_name', 'Unknown Client')
        
        prompt = f"""
        Summarize the following tasks for a project. Provide a concise summary that highlights:
        1. Main activities and work areas
        2. Key deliverables or outcomes
        3. Types of work performed
        4. Overall contribution to the project

        Project: {project_name}
        Client: {client_name}
        Total Hours: {total_hours}

        Tasks:
        {all_tasks}

        Provide a clear, professional summary in 1 paragraph:
        """
        
        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            return summary
            
        except Exception as e:
            st.error(f"Error generating summary with Gemini: {str(e)}")
            return f"Error generating summary: {str(e)}"
   
    def save_task_summary(self, employee_code: str, project_id: str, summary_text: str, 
                         project_data: Dict, model_used: str, generated_by: str = None) -> bool:
        """Save task summary to database"""
        conn = self.get_database_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Helper function to convert dates to strings
            def convert_date_to_string(date_obj):
                if date_obj is None:
                    return None
                if isinstance(date_obj, (datetime, date)):
                    return date_obj.isoformat()
                return str(date_obj)
            
            # Convert date objects to strings for database insertion
            start_date = convert_date_to_string(project_data.get('start_date'))
            end_date = convert_date_to_string(project_data.get('end_date'))
            
            # Prepare metadata with proper date handling
            tasks_for_metadata = project_data.get('tasks', [])
            # Convert any date objects in tasks to strings
            serializable_tasks = []
            for task in tasks_for_metadata:
                if isinstance(task, dict):
                    serializable_task = {}
                    for key, value in task.items():
                        if isinstance(value, (datetime, date)):
                            serializable_task[key] = value.isoformat()
                        else:
                            serializable_task[key] = value
                    serializable_tasks.append(serializable_task)
                else:
                    serializable_tasks.append(task)
            
            metadata = {
                'detailed_tasks': serializable_tasks,
                'project_name': project_data.get('project_name'),
                'client_name': project_data.get('client_name'),
                'generation_parameters': {
                    'model_used': model_used,
                    'generated_at': datetime.now().isoformat()
                }
            }
            
            # Check if summary already exists
            check_query = """
            SELECT summary_id, version FROM task_summary 
            WHERE employee_code = %s AND project_id = %s AND status = 'Active'
            ORDER BY version DESC LIMIT 1
            """
            cursor.execute(check_query, (employee_code, project_id))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing summary (create new version)
                summary_id, current_version = existing
                new_version = current_version + 1
                
                # Archive old version
                cursor.execute(
                    "UPDATE task_summary SET status = 'Archived' WHERE summary_id = %s",
                    (summary_id,)
                )
                
                # Insert new version
                insert_query = """
                INSERT INTO task_summary (
                    employee_code, project_id, summary_text, summary_type, model_used,
                    total_hours, task_count, date_range_start, date_range_end,
                    generated_by, version, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING summary_id
                """
            else:
                # Insert new summary
                new_version = 1
                insert_query = """
                INSERT INTO task_summary (
                    employee_code, project_id, summary_text, summary_type, model_used,
                    total_hours, task_count, date_range_start, date_range_end,
                    generated_by, version, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING summary_id
                """
            
            cursor.execute(insert_query, (
                employee_code,
                project_id,
                summary_text,
                'AI_GENERATED',
                model_used,
                float(project_data.get('total_hours', 0)),
                len(project_data.get('tasks', [])),
                start_date,  # Now properly converted to string
                end_date,    # Now properly converted to string
                generated_by,
                new_version,
                json.dumps(metadata)  # metadata now contains only JSON-serializable objects
            ))
            
            new_summary_id = cursor.fetchone()[0]
            
            # If updating, record in history
            if existing:
                history_query = """
                INSERT INTO task_summary_history (
                    summary_id, old_summary_text, new_summary_text, 
                    old_version, new_version, change_reason, changed_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(history_query, (
                    summary_id,
                    None,  # We could fetch the old text if needed
                    summary_text,
                    current_version,
                    new_version,
                    'AI regeneration',
                    generated_by
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Error saving task summary: {str(e)}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def get_saved_summaries(self, employee_code: str = None, project_id: str = None) -> pd.DataFrame:
        """Retrieve saved task summaries"""
        conn = self.get_database_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            base_query = """
            SELECT 
                ts.summary_id,
                ts.employee_code,
                e.employee_name,
                ts.project_id,
                p.project_name,
                p.client_name,
                ts.summary_text,
                ts.model_used,
                ts.total_hours,
                ts.task_count,
                ts.date_range_start,
                ts.date_range_end,
                ts.generated_at,
                ts.version,
                ts.status
            FROM task_summary ts
            LEFT JOIN employee e ON ts.employee_code = e.employee_code
            LEFT JOIN project p ON ts.project_id = p.project_id
            WHERE ts.status = 'Active'
            """
            
            params = []
            if employee_code:
                base_query += " AND ts.employee_code = %s"
                params.append(employee_code)
            
            if project_id:
                base_query += " AND ts.project_id = %s"
                params.append(project_id)
            
            base_query += " ORDER BY ts.generated_at DESC"
            
            df = pd.read_sql_query(base_query, conn, params=params)
            conn.close()
            return df
            
        except Exception as e:
            st.error(f"Error retrieving saved summaries: {str(e)}")
            if conn:
                conn.close()
            return pd.DataFrame()
    
    def delete_summary(self, summary_id: int) -> bool:
        """Soft delete a task summary"""
        conn = self.get_database_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE task_summary SET status = 'Deleted' WHERE summary_id = %s",
                (summary_id,)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Error deleting summary: {str(e)}")
            if conn:
                conn.close()
            return False
    
    def get_employee_summary_stats(self, employee_code: str) -> Dict:
        """Get employee summary statistics"""
        conn = self.get_database_connection()
        if not conn:
            return {}
        
        try:
            query = """
            SELECT 
                e.employee_code,
                e.employee_name,
                d.department_name,
                des.designation_name,
                COUNT(DISTINCT t.project_id) as total_projects,
                SUM(t.hours_worked) as total_hours,
                COUNT(t.timesheet_id) as total_entries,
                MIN(t.work_date) as first_entry,
                MAX(t.work_date) as last_entry,
                COUNT(DISTINCT t.task_description) as unique_tasks
            FROM employee e
            LEFT JOIN timesheet t ON e.employee_code = t.employee_code
            LEFT JOIN department d ON e.department_id = d.department_id
            LEFT JOIN designation des ON e.designation_id = des.designation_id
            WHERE e.employee_code = %s
            GROUP BY e.employee_code, e.employee_name, d.department_name, des.designation_name
            """
            
            df = pd.read_sql_query(query, conn, params=[employee_code])
            conn.close()
            
            if df.empty:
                return {}
            
            return df.iloc[0].to_dict()
            
        except Exception as e:
            st.error(f"Error fetching employee stats: {str(e)}")
            if conn:
                conn.close()
            return {}
    
    def get_all_employees(self) -> List[str]:
        """Get list of all employees who have timesheet entries"""
        conn = self.get_database_connection()
        if not conn:
            return []
        
        try:
            query = """
            SELECT DISTINCT e.employee_code, e.employee_name
            FROM employee e
            INNER JOIN timesheet t ON e.employee_code = t.employee_code
            ORDER BY e.employee_name
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Return list of formatted strings: "CODE - Name"
            return [f"{row['employee_code']} - {row['employee_name']}" for _, row in df.iterrows()]
        
        except Exception as e:
            st.error(f"Error fetching employees: {str(e)}")
            if conn:
                conn.close()
            return []

def task_summarizer():
    """Enhanced Tab 5: Employee Task Summarizer with Database Storage"""
    
    st.header("Employee Task Summarizer")
    
    # Check for required environment variables
    required_vars = ['GEMINI_API_KEY', 'DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please set the following environment variables:")
        for var in missing_vars:
            st.code(f"export {var}=your_value_here")
        return
    
    # Initialize the task summarizer
    try:
        summarizer = TaskSummarizer()
        if not hasattr(summarizer, 'model'):
            return  # Error already shown in __init__
    except Exception as e:
        st.error(f"Error initializing Task Summarizer: {str(e)}")
        return
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Generate Summaries", "Manage Saved Summaries"])
    
    with tab1:
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Employee Selection")
            
            # Get all employees
            employees = summarizer.get_all_employees()
            
            if not employees:
                st.error("No employees found with timesheet entries or connection error.")
                return
            
            # Employee selection
            selected_employee_display = st.selectbox(
                "Select Employee:",
                options=employees,
                help="Choose an employee to summarize their project tasks",
                key="employee_select_tab5"
            )
            
            # Extract employee code from display string
            selected_employee = selected_employee_display.split(' - ')[0] if selected_employee_display else None
            
            # Model selection
            model_options = {
                'gemini-1.5-flash': 'Gemini 1.5 Flash (Fast & Efficient)',
                'gemini-1.5-flash-002': 'Gemini 1.5 Flash-002 (Latest)',
            }
            
            selected_model = st.selectbox(
                "Select AI Model:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                help="Choose the Gemini model for task summarization",
                key="model_select_tab5"
            )
            
            # Summarize tasks button
            if st.button("🔍 Generate Summaries", use_container_width=True, key="summarize_btn"):
                if selected_employee:
                    with st.spinner("Analyzing timesheet data and generating summaries..."):
                        # Update model based on selection
                        summarizer.model = genai.GenerativeModel(selected_model)
                        
                        # Store results in session state with tab-specific keys
                        st.session_state.tab5_current_employee = selected_employee
                        st.session_state.tab5_employee_stats = summarizer.get_employee_summary_stats(selected_employee)
                        st.session_state.tab5_projects_data = summarizer.fetch_employee_timesheet_data(selected_employee)
                        
                        # Generate summaries for each project
                        st.session_state.tab5_project_summaries = {}
                        for project_id, project_data in st.session_state.tab5_projects_data.items():
                            st.session_state.tab5_project_summaries[project_id] = summarizer.summarize_project_tasks_with_gemini(project_data)
                        
                        st.session_state.tab5_summary_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.tab5_model_used = selected_model
                        
                        st.success("Task summaries generated successfully!")
        
        with col2:
            st.subheader("Generated Summaries")
            
            # Display results if available
            if hasattr(st.session_state, 'tab5_current_employee') and st.session_state.tab5_current_employee:
                
                # Employee info
                if st.session_state.tab5_employee_stats:
                    stats = st.session_state.tab5_employee_stats
                    
                    # Employee information
                    st.info(f"**{stats.get('employee_name', 'Unknown')}** ({st.session_state.tab5_current_employee})")
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"**Department:** {stats.get('department_name', 'N/A')}")
                    with col_info2:
                        st.write(f"**Designation:** {stats.get('designation_name', 'N/A')}")
                    
                    # Employee statistics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Hours", f"{stats.get('total_hours', 0) or 0:.1f}")
                    with col_b:
                        st.metric("Projects", stats.get('total_projects', 0) or 0)
                    with col_c:
                        st.metric("Entries", stats.get('total_entries', 0) or 0)
                    
                    st.divider()
                    
                    # Project summaries section
                    if st.session_state.tab5_projects_data and st.session_state.tab5_project_summaries:
                        
                        # Save all summaries button
                        col_save1, col_save2 = st.columns(2)
                        with col_save1:
                            if st.button("💾 Save All Summaries to Database", use_container_width=True, key="save_all_btn"):
                                with st.spinner("Saving summaries to database..."):
                                    success_count = 0
                                    total_count = len(st.session_state.tab5_project_summaries)
                                    
                                    for project_id, summary_text in st.session_state.tab5_project_summaries.items():
                                        project_data = st.session_state.tab5_projects_data[project_id]
                                        if summarizer.save_task_summary(
                                            st.session_state.tab5_current_employee,
                                            project_id,
                                            summary_text,
                                            project_data,
                                            st.session_state.tab5_model_used,
                                            st.session_state.tab5_current_employee  # Using employee as generated_by
                                        ):
                                            success_count += 1
                                    
                                    if success_count == total_count:
                                        st.success(f"✅ All {success_count} summaries saved successfully!")
                                    else:
                                        st.warning(f"⚠️ {success_count}/{total_count} summaries saved successfully.")
                        
                        with col_save2:
                            # Check for existing summaries
                            existing_summaries = summarizer.get_saved_summaries(employee_code=st.session_state.tab5_current_employee)
                            if not existing_summaries.empty:
                                st.info(f"📄 {len(existing_summaries)} existing summaries found")
                        
                        st.divider()
                        
                        # Display individual project summaries
                        for project_id, project_data in st.session_state.tab5_projects_data.items():
                            
                            # Project header with save button
                            col_header, col_save_individual = st.columns([3, 1])
                            
                            with col_header:
                                st.markdown(f"### {project_data.get('project_name', project_id)}")
                            
                            with col_save_individual:
                                if st.button(f"💾 Save", key=f"save_{project_id}", help=f"Save summary for {project_id}"):
                                    summary_text = st.session_state.tab5_project_summaries.get(project_id, "")
                                    if summarizer.save_task_summary(
                                        st.session_state.tab5_current_employee,
                                        project_id,
                                        summary_text,
                                        project_data,
                                        st.session_state.tab5_model_used,
                                        st.session_state.tab5_current_employee
                                    ):
                                        st.success("✅ Saved!")
                                    else:
                                        st.error("❌ Failed to save")
                            
                            # Project details
                            container = st.container()
                            with container:
                                col_x, col_y = st.columns(2)
                                with col_x:
                                    st.write(f"**Project ID:** {project_id}")
                                    st.write(f"**Client:** {project_data.get('client_name', 'N/A')}")
                                with col_y:
                                    st.write(f"**Total Hours:** {project_data.get('total_hours', 0):.1f}")
                                    st.write(f"**Tasks:** {len(project_data.get('tasks', []))}")
                                
                                # AI-generated summary (editable)
                                st.markdown("**AI Summary:**")
                                edited_summary = st.text_area(
                                    f"Edit summary for {project_data.get('project_name', project_id)}:",
                                    value=st.session_state.tab5_project_summaries.get(project_id, ""),
                                    height=100,
                                    key=f"summary_edit_{project_id}",
                                    help="You can edit the AI-generated summary before saving"
                                )
                                
                                # Update the summary in session state if edited
                                if edited_summary != st.session_state.tab5_project_summaries.get(project_id, ""):
                                    st.session_state.tab5_project_summaries[project_id] = edited_summary
                                
                                # Show detailed tasks using toggle
                                show_details = st.toggle(
                                    f"Show Detailed Tasks",
                                    key=f"show_tasks_{project_id}",
                                    help=f"Toggle to show/hide detailed task breakdown"
                                )
                                
                                if show_details:
                                    st.markdown("**Detailed Tasks:**")
                                    tasks_df = pd.DataFrame(project_data.get('tasks', []))
                                    if not tasks_df.empty:
                                        tasks_df = tasks_df.sort_values('date', ascending=False)
                                        st.dataframe(
                                            tasks_df,
                                            use_container_width=True,
                                            hide_index=True,
                                            column_config={
                                                "description": "Task Description",
                                                "hours": st.column_config.NumberColumn("Hours", format="%.2f"),
                                                "date": st.column_config.DateColumn("Date")
                                            }
                                        )
                            
                            st.divider()
                    
                    else:
                        st.warning("No timesheet data found for this employee.")
                
                else:
                    st.error("Employee not found in database.")
            
            else:
                st.info("👈 Select an employee and click 'Generate Summaries' to see results.")
    
    with tab2:
        st.subheader("Manage Saved Summaries")
        
        # Filters
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            # Employee filter
            employees = summarizer.get_all_employees()
            selected_filter_employee = st.selectbox(
                "Filter by Employee:",
                options=["All"] + employees,
                key="filter_employee"
            )
            
            filter_employee_code = None
            if selected_filter_employee != "All":
                filter_employee_code = selected_filter_employee.split(' - ')[0]
        
        with col_filter2:
            # Project filter - could be enhanced to get project list from DB
            filter_project = st.text_input("Filter by Project ID:", key="filter_project")
        
        with col_filter3:
            if st.button("🔍 Load Summaries", use_container_width=True):
                st.session_state.saved_summaries_df = summarizer.get_saved_summaries(
                    employee_code=filter_employee_code,
                    project_id=filter_project if filter_project else None
                )
        
        # Display saved summaries
        if hasattr(st.session_state, 'saved_summaries_df') and not st.session_state.saved_summaries_df.empty:
            df = st.session_state.saved_summaries_df
            
            st.write(f"**Found {len(df)} saved summaries:**")
            
            # Display summaries
            for idx, row in df.iterrows():
                with st.expander(f"{row['employee_name']} - {row['project_name']} (v{row['version']})"):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**Employee:** {row['employee_name']} ({row['employee_code']})")
                        st.write(f"**Project:** {row['project_name']}")
                        st.write(f"**Client:** {row['client_name']}")
                        st.write(f"**Model Used:** {row['model_used']}")
                    
                    with col_info2:
                        st.write(f"**Total Hours:** {row['total_hours']:.1f}")
                        st.write(f"**Task Count:** {row['task_count']}")
                        st.write(f"**Generated:** {row['generated_at'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Version:** {row['version']}")
                    
                    st.markdown("**Summary:**")
                    st.write(row['summary_text'])
                    
                    # # Action buttons
                    col_action1, col_action2, col_action3 = st.columns(3)
                    
                    # with col_action1:
                    #     if st.button(" Delete", key=f"delete_{row['summary_id']}", help="Delete this summary"):
                    #         if summarizer.delete_summary(row['summary_id']):
                    #             st.success("Summary deleted successfully!")
                    #             st.rerun()
                    #         else:
                    #             st.error("Failed to delete summary")
                    
                    # with col_action2:
                    #     # Download individual summary
                    #     summary_data = {
                    #         "employee_code": row['employee_code'],
                    #         "employee_name": row['employee_name'],
                    #         "project_id": row['project_id'],
                    #         "project_name": row['project_name'],
                    #         "client_name": row['client_name'],
                    #         "summary_text": row['summary_text'],
                    #         "total_hours": float(row['total_hours']),
                    #         "task_count": row['task_count'],
                    #         "generated_at": row['generated_at'].isoformat(),
                    #         "model_used": row['model_used'],
                    #         "version": row['version']
                    #     }
                        
                    #     st.download_button(
                    #         label=" Download JSON",
                    #         data=json.dumps(summary_data, indent=2, default=str),
                    #         file_name=f"summary_{row['employee_code']}_{row['project_id']}_v{row['version']}.json",
                    #         mime="application/json",
                    #         key=f"download_{row['summary_id']}"
                    #     )
                    
                    with col_action3:
                        # Copy to clipboard (text format)
                        summary_text = f"""
TASK SUMMARY
Employee: {row['employee_name']} ({row['employee_code']})
Project: {row['project_name']} ({row['project_id']})
Client: {row['client_name']}
Total Hours: {row['total_hours']:.1f}
Generated: {row['generated_at'].strftime('%Y-%m-%d %H:%M')}

Summary:
{row['summary_text']}
                        """.strip()
                        
                        if st.button("Copy Text", key=f"copy_{row['summary_id']}", help="Copy summary as text"):
                            st.code(summary_text, language=None)
        
        else:
            if hasattr(st.session_state, 'saved_summaries_df'):
                st.info("No saved summaries found matching the filters.")
            else:
                st.info("Click 'Load Summaries' to view saved summaries.")
        
        # Bulk operations
        if hasattr(st.session_state, 'saved_summaries_df') and not st.session_state.saved_summaries_df.empty:
            st.divider()
            st.subheader("Bulk Operations")
            
            col_bulk1, col_bulk2 = st.columns(2)
            
            with col_bulk1:
                # Export all summaries
                if st.button("📤 Export All Summaries", use_container_width=True):
                    df = st.session_state.saved_summaries_df
                    
                    # Create comprehensive export
                    export_data = {
                        "export_date": datetime.now().isoformat(),
                        "total_summaries": len(df),
                        "summaries": []
                    }
                    
                    for _, row in df.iterrows():
                        summary_data = {
                            "summary_id": row['summary_id'],
                            "employee_code": row['employee_code'],
                            "employee_name": row['employee_name'],
                            "project_id": row['project_id'],
                            "project_name": row['project_name'],
                            "client_name": row['client_name'],
                            "summary_text": row['summary_text'],
                            "total_hours": float(row['total_hours']),
                            "task_count": row['task_count'],
                            "date_range_start": row['date_range_start'].isoformat() if row['date_range_start'] else None,
                            "date_range_end": row['date_range_end'].isoformat() if row['date_range_end'] else None,
                            "generated_at": row['generated_at'].isoformat(),
                            "model_used": row['model_used'],
                            "version": row['version'],
                            "status": row['status']
                        }
                        export_data["summaries"].append(summary_data)
                    
                    st.download_button(
                        label="📥 Download All Summaries (JSON)",
                        data=json.dumps(export_data, indent=2, default=str),
                        file_name=f"all_task_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="export_all_json"
                    )
            
            with col_bulk2:
                # Summary statistics
                df = st.session_state.saved_summaries_df
                
                st.markdown("**Summary Statistics:**")
                st.write(f"• Total Summaries: {len(df)}")
                st.write(f"• Unique Employees: {df['employee_code'].nunique()}")
                st.write(f"• Unique Projects: {df['project_id'].nunique()}")
                st.write(f"• Total Hours Covered: {df['total_hours'].sum():.1f}")
                st.write(f"• Average Hours per Project: {df['total_hours'].mean():.1f}")
                
                # Most recent summary
                latest = df.loc[df['generated_at'].idxmax()]
                st.write(f"• Latest Summary: {latest['employee_name']} - {latest['project_name']}")

# Additional utility functions for database management
def create_database_tables():
    """Create the task summary tables if they don't exist"""
    summarizer = TaskSummarizer()
    conn = summarizer.get_database_connection()
    
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Read and execute the SQL from the first artifact
        sql_commands = """
        -- TASK_SUMMARY table to store AI-generated summaries
        CREATE TABLE IF NOT EXISTS task_summary (
            summary_id SERIAL PRIMARY KEY,
            employee_code VARCHAR(20) NOT NULL REFERENCES employee(employee_code),
            project_id VARCHAR(50) NOT NULL REFERENCES project(project_id),
            summary_text TEXT NOT NULL,
            summary_type VARCHAR(50) DEFAULT 'AI_GENERATED',
            model_used VARCHAR(100),
            total_hours DECIMAL(8,2),
            task_count INTEGER,
            date_range_start DATE,
            date_range_end DATE,
            generated_by VARCHAR(20) REFERENCES employee(employee_code),
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            version INTEGER DEFAULT 1,
            status VARCHAR(20) DEFAULT 'Active',
            metadata JSONB,
            CONSTRAINT unique_employee_project_version UNIQUE(employee_code, project_id, version),
            CONSTRAINT chk_date_range CHECK (date_range_end IS NULL OR date_range_end >= date_range_start)
        );

        CREATE INDEX IF NOT EXISTS idx_task_summary_employee_project ON task_summary(employee_code, project_id);
        CREATE INDEX IF NOT EXISTS idx_task_summary_generated_at ON task_summary(generated_at);
        CREATE INDEX IF NOT EXISTS idx_task_summary_status ON task_summary(status);

        CREATE TABLE IF NOT EXISTS task_summary_history (
            history_id SERIAL PRIMARY KEY,
            summary_id INTEGER NOT NULL REFERENCES task_summary(summary_id),
            old_summary_text TEXT,
            new_summary_text TEXT,
            old_version INTEGER,
            new_version INTEGER,
            change_reason TEXT,
            changed_by VARCHAR(20) REFERENCES employee(employee_code),
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(sql_commands)
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error creating tables: {str(e)}")
        if conn:
            conn.close()
        return False

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Enhanced Task Summarizer",
        page_icon="📋",
        layout="wide"
    )
    
    # Add a setup section
    with st.sidebar:
        st.header("Setup")
        if st.button("🔧 Create Database Tables"):
            if create_database_tables():
                st.success("Database tables created successfully!")
            else:
                st.error("Failed to create database tables")
        
        st.divider()
        st.info("Make sure to set the following environment variables:")
        st.code("""
export GEMINI_API_KEY=your_gemini_api_key
export DB_HOST=localhost
export DB_NAME=your_database_name
export DB_USER=your_db_user
export DB_PASSWORD=your_db_password
export DB_PORT=5432
        """)
    
    task_summarizer()