import streamlit as st
import pandas as pd
from datetime import datetime, date
from logs.activity_logger import get_logger
from sqlalchemy import text
import time


def render_allocations(engine):
    """Render the allocations management page"""
    st.subheader("Project & Resource Allocations Management")
    
    # Get the global activity logger instance
    logger = get_logger()

    # Create tabs for project management
    tab1, tab2, tab3 = st.tabs(["Project List", "Edit Project", "Manage Allocations"])
    
    with tab1:
        render_projects_list(engine, logger)
    
    with tab2:
        render_edit_project(engine, logger)

    with tab3:
        render_manage_allocations(engine, logger)


def get_employees_list(engine):
    """Get list of all active employees"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT employee_code, employee_name, department_name
                FROM employee
                WHERE status = 'Active'
                ORDER BY employee_name
            """))
            return result.fetchall()
    except Exception as e:
        st.error(f"Error fetching employees: {e}")
        return []


def get_employee_details(engine, employee_code):
    """Get employee basic details"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT employee_code, employee_name, department_name, email, mobile_number
                FROM employee
                WHERE employee_code = :employee_code AND status = 'Active'
            """), {"employee_code": employee_code})
            return result.fetchone()
    except Exception as e:
        st.error(f"Error fetching employee details: {e}")
        return None


def get_employee_allocations(engine, employee_code):
    """Get current active allocations for an employee"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT
                    pa.allocation_id,
                    pa.project_id,
                    p.project_name,
                    pa.allocation_percentage,
                    p.start_date,
                    p.end_date,
                    pa.effective_from,
                    pa.effective_to,
                    COALESCE(
                        (SELECT SUM(t.hours_worked) 
                         FROM timesheet t 
                         WHERE t.employee_code = pa.employee_code 
                         AND t.project_id = pa.project_id), 0
                    ) as total_hours,
                    COALESCE(
                        (SELECT COUNT(DISTINCT t.work_date) 
                         FROM timesheet t 
                         WHERE t.employee_code = pa.employee_code 
                         AND t.project_id = pa.project_id), 0
                    ) as days_worked,
                    COALESCE(
                        (SELECT MIN(t.work_date) 
                         FROM timesheet t 
                         WHERE t.employee_code = pa.employee_code 
                         AND t.project_id = pa.project_id), p.start_date
                    ) as first_day,
                    COALESCE(
                        (SELECT MAX(t.work_date) 
                         FROM timesheet t 
                         WHERE t.employee_code = pa.employee_code 
                         AND t.project_id = pa.project_id), p.end_date
                    ) as last_day
                FROM project_allocation pa
                JOIN project p ON pa.project_id = p.project_id
                WHERE pa.employee_code = :employee_code 
                AND pa.status = 'Active'
                AND pa.allocation_id IN (
                    SELECT MAX(allocation_id) 
                    FROM project_allocation 
                    WHERE employee_code = :employee_code 
                    AND status = 'Active'
                    GROUP BY project_id
                )
                ORDER BY pa.project_id
            """), {"employee_code": employee_code})
            return result.fetchall()
    except Exception as e:
        st.error(f"Error fetching allocations: {e}")
        return []


def get_valid_created_by(engine):
    """Get a valid employee code to use as created_by"""
    try:
        with engine.connect() as conn:
            # Try to get the current user from session state
            current_user = st.session_state.get('user')
            if current_user:
                result = conn.execute(text("""
                    SELECT employee_code FROM employee 
                    WHERE employee_code = :employee_code AND status = 'Active'
                """), {"employee_code": current_user})
                if result.fetchone():
                    return current_user

            # If no valid current user, get the first active employee
            result = conn.execute(text("""
                SELECT employee_code FROM employee 
                WHERE status = 'Active' 
                ORDER BY employee_code 
                LIMIT 1
            """))
            row = result.fetchone()
            return row[0] if row else None
    except Exception as e:
        st.error(f"Error getting valid created_by: {e}")
        return None


def display_employee_details(employee_details, allocations):
    """Display employee details card"""
    emp_code, emp_name, dept_name, email, mobile = employee_details

    # Calculate total allocation
    total_allocation = sum(allocation[3] for allocation in allocations) if allocations else 0

    st.markdown(f"""
    **{emp_name} ({emp_code}) - {dept_name}**

    **Email:** {email or 'N/A'}  
    **Mobile Number:** {mobile or 'N/A'}  
    **Total Allocation:** {total_allocation:.1f}%
    """)

    #if total_allocation > 100:
    #    st.error(f"Warning: Total allocation ({total_allocation:.1f}%) exceeds 100%")
    #elif total_allocation == 0:
    #    st.info("No active allocations")


def display_allocations_table(engine, allocations, employee_code, logger):
    """Display allocations table with clickable allocation percentages for editing"""
    st.markdown("**Project Assignments with Allocation:**")

    # Create columns for the table header (removed Edit column)
    col_headers = st.columns([1, 2.5, 1, 1, 1.2, 1.2, 1])
    headers = ["Project", "Project Name", "Hours", "Days", "First Day", "Last Day", "Allocation (%)"]

    for i, header in enumerate(headers):
        col_headers[i].markdown(f"**{header}**")

    st.markdown("---")

    # Display each allocation row
    for allocation in allocations:
        allocation_id, project_id, project_name, allocation_percentage, start_date, end_date, effective_from, effective_to, total_hours, days_worked, first_day, last_day = allocation

        cols = st.columns([1, 2.5, 1, 1, 1.2, 1.2, 1])

        with cols[0]:
            st.write(project_id)
        with cols[1]:
            st.write(project_name)
        with cols[2]:
            st.write(f"{total_hours:.1f}")
        with cols[3]:
            st.write(str(days_worked))
        with cols[4]:
            st.write(str(first_day) if first_day else "N/A")
        with cols[5]:
            st.write(str(last_day) if last_day else "N/A")
        with cols[6]:
            # Check if this allocation is being edited
            edit_key = f"edit_{allocation_id}"
            if st.session_state.get(edit_key, False):
                # Show edit form as number input
                new_percentage = st.number_input(
                    "",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(allocation_percentage),
                    step=0.1,
                    key=f"new_percentage_{allocation_id}"
                )
                # Auto-store the change when value changes
                if new_percentage != float(allocation_percentage):
                    st.session_state[f"pending_change_{allocation_id}"] = {
                        'new_percentage': new_percentage,
                        'allocation_id': allocation_id,
                        'employee_code': employee_code,
                        'old_percentage': float(allocation_percentage)
                    }
            else:
                # Show clickable allocation percentage
                if st.button(f"{allocation_percentage:.1f}%", key=f"allocation_btn_{allocation_id}",
                             help="Click to edit this allocation"):
                    st.session_state[edit_key] = True
                    st.rerun()


def display_save_changes_section(engine, employee_code, logger):
    """Display save changes section below all allocation details"""
    st.markdown("---")
    st.markdown("### Save Changes")

    # Check if there are any pending changes
    pending_changes = {}
    for key in st.session_state.keys():
        if key.startswith("pending_change_"):
            allocation_id = key.replace("pending_change_", "")
            pending_changes[allocation_id] = st.session_state[key]

    if pending_changes:
        st.markdown("**Pending Changes:**")
        for allocation_id, change_data in pending_changes.items():
            old_pct = change_data.get('old_percentage', 0)
            new_pct = change_data['new_percentage']
            st.write(f"- Allocation ID {allocation_id}: {old_pct:.1f}% → {new_pct:.1f}%")

        # Reason input for all changes
        change_reason = st.text_area(
            "Change Reason (Required for all changes):",
            key="bulk_change_reason",
            placeholder="Enter reason for allocation changes...",
            height=100
        )

        col_confirm, col_cancel_all = st.columns(2)

        with col_confirm:
            if st.button("Confirm All Changes", key="confirm_all_changes"):
                if change_reason.strip():
                    all_success = True
                    for allocation_id, change_data in pending_changes.items():
                        success = update_allocation(
                            engine,
                            change_data['allocation_id'],
                            change_data['employee_code'],
                            change_data['new_percentage'],
                            'Active',  # Status remains active for bulk changes
                            datetime.now().date(),  # Effective from today
                            None,  # No end date for bulk changes
                            change_reason.strip(),
                            logger
                        )
                        if not success:
                            all_success = False

                    if all_success:
                        st.success("All allocations updated successfully!")
                        # Clear all pending changes and edit states
                        for allocation_id in pending_changes.keys():
                            st.session_state.pop(f"edit_{allocation_id}", None)
                            st.session_state.pop(f"pending_change_{allocation_id}", None)
                        st.session_state.pop("bulk_change_reason", None)
                        st.rerun()
                    else:
                        st.error("Some allocations failed to update. Please check the logs.")
                else:
                    st.error("Change reason is required for all changes")

        with col_cancel_all:
            if st.button("Cancel All Changes", key="cancel_all_changes"):
                # Clear all pending changes and edit states
                for allocation_id in pending_changes.keys():
                    st.session_state.pop(f"edit_{allocation_id}", None)
                    st.session_state.pop(f"pending_change_{allocation_id}", None)
                st.session_state.pop("bulk_change_reason", None)
                st.rerun()
    else:
        st.info("No pending changes to save")


def update_allocation(engine, old_allocation_id, employee_code, new_percentage, new_status, new_effective_from, new_effective_to, change_reason, logger):
    """Update allocation by creating new record and deactivating old one"""
    try:
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()

            try:
                # Get current allocation details
                result = conn.execute(text("""
                    SELECT project_id, effective_from, effective_to, allocation_percentage
                    FROM project_allocation
                    WHERE allocation_id = :allocation_id
                """), {"allocation_id": old_allocation_id})

                allocation_data = result.fetchone()
                if not allocation_data:
                    st.error("Allocation not found")
                    return False

                project_id, effective_from, effective_to, old_percentage = allocation_data
                
                # Get a valid created_by employee code
                created_by = get_valid_created_by(engine)
                if not created_by:
                    st.error("No valid employee found for created_by field")
                    return False

                # Deactivate old allocation
                conn.execute(text("""
                    UPDATE project_allocation
                    SET status = 'Inactive'
                    WHERE allocation_id = :allocation_id
                """), {"allocation_id": old_allocation_id})

                # Insert new allocation record
                conn.execute(text("""
                    INSERT INTO project_allocation 
                    (employee_code, project_id, allocation_percentage, effective_from, 
                     effective_to, status, created_by, change_reason, created_at)
                    VALUES (:employee_code, :project_id, :allocation_percentage, :effective_from, 
                            :effective_to, :status, :created_by, :change_reason, :created_at)
                """), {
                    "employee_code": employee_code,
                    "project_id": project_id,
                    "allocation_percentage": new_percentage,
                    "effective_from": new_effective_from,
                    "effective_to": new_effective_to,
                    "status": new_status,
                    "created_by": created_by,
                    "change_reason": change_reason,
                    "created_at": datetime.now()
                })

                # Commit transaction
                trans.commit()

                # Log the allocation update
                logger.log_event(
                    event_type="ALLOCATION_UPDATE",
                    description=f"Updated allocation for {employee_code} on project {project_id}",
                    user=st.session_state.get('username', created_by),
                    details={
                        "employee_code": employee_code,
                        "project_id": project_id,
                        "old_percentage": float(old_percentage),
                        "new_percentage": new_percentage,
                        "user_full_name": st.session_state.get('user_full_name', 'Unknown'),
                        "timestamp": str(datetime.now()),
                        "change_reason": change_reason
                    }
                )

                return True

            except Exception as e:
                trans.rollback()
                raise e

    except Exception as e:
        st.error(f"Error updating allocation: {e}")

        # Log the error
        logger.log_event(
            event_type="ALLOCATION_UPDATE_ERROR",
            description=f"Failed to update allocation: {str(e)}",
            user=st.session_state.get('username', 'system'),
            details={
                "error": str(e), 
                "allocation_id": old_allocation_id,
                "user_full_name": st.session_state.get('user_full_name', 'Unknown'),
                "timestamp": str(datetime.now())
            }
        )

        return False


def validate_total_allocation(engine, employee_code, exclude_allocation_id=None):
    """Validate that total allocation doesn't exceed 100%"""
    try:
        with engine.connect() as conn:
            if exclude_allocation_id:
                result = conn.execute(text("""
                    SELECT SUM(allocation_percentage)
                    FROM project_allocation
                    WHERE employee_code = :employee_code AND status = 'Active'
                    AND allocation_id != :exclude_allocation_id
                """), {
                    "employee_code": employee_code,
                    "exclude_allocation_id": exclude_allocation_id
                })
            else:
                result = conn.execute(text("""
                    SELECT SUM(allocation_percentage)
                    FROM project_allocation
                    WHERE employee_code = :employee_code AND status = 'Active'
                """), {"employee_code": employee_code})

            row = result.fetchone()
            return float(row[0]) if row[0] else 0.0

    except Exception as e:
        st.error(f"Error validating allocation: {e}")
        return 0.0


def render_projects_list(engine, logger):
    """Display list of all projects with key metrics"""
    st.header("Projects List")
    
    # Add new project section
    with st.expander("Add New Project", expanded=False):
        with st.form("add_project_form"):
            col1, col2 = st.columns(2)
            with col1:
                project_id = st.text_input("Project ID", placeholder="Enter unique project ID")
                project_name = st.text_input("Project Name", placeholder="Enter project name")
                client_name = st.text_input("Client Name", placeholder="Enter client name")
            
            with col2:
                # Get managers list for selection
                managers_df = pd.read_sql(text("""
                    SELECT e.employee_code, e.employee_name, e.department_name
                    FROM employee e
                    WHERE e.status = 'Active'
                    ORDER BY e.employee_name
                """), engine)
                
                manager = st.selectbox(
                    "Project Manager",
                    options=managers_df['employee_code'].tolist(),
                    format_func=lambda x: f"{managers_df[managers_df['employee_code']==x]['employee_name'].iloc[0]} ({managers_df[managers_df['employee_code']==x]['department_name'].iloc[0]})"
                )
                
                start_date = st.date_input("Start Date", min_value=date.today())
                end_date = st.date_input("End Date", min_value=start_date)
                status = st.selectbox("Status", ["Active", "Inactive", "Completed"])
            
            submit = st.form_submit_button("Add Project")
            
            if submit and project_id and project_name:
                try:
                    with engine.connect() as conn:
                        # Start transaction
                        trans = conn.begin()
                        try:
                            # Check if project ID already exists
                            result = conn.execute(text("""
                                SELECT COUNT(*) FROM project WHERE project_id = :project_id
                            """), {"project_id": project_id})
                            
                            if result.scalar() > 0:
                                st.error(f"Project ID '{project_id}' already exists. Please use a unique ID.")
                                return

                            # Insert the new project
                            conn.execute(text("""
                                INSERT INTO project (project_id, project_name, client_name, manager_id, status, start_date, end_date)
                                VALUES (:project_id, :project_name, :client_name, :manager_id, :status, :start_date, :end_date)
                            """), {
                                "project_id": project_id,
                                "project_name": project_name,
                                "client_name": client_name,
                                "manager_id": manager,
                                "status": status,
                                "start_date": start_date,
                                "end_date": end_date
                            })
                            
                            # Commit the transaction
                            trans.commit()
                            
                            logger.log_event(
                                event_type="PROJECT_CREATE",
                                description=f"Created new project {project_name}",
                                user=st.session_state.get('user', 'system'),
                                details={
                                    'project_id': project_id,
                                    'project_name': project_name,
                                    'client_name': client_name,
                                    'manager_id': manager,
                                    'status': status,
                                    'start_date': str(start_date),
                                    'end_date': str(end_date)
                                }
                            )
                            st.success("Project added successfully!")
                            time.sleep(0.1)  # Small delay to ensure the database transaction is complete
                            st.rerun()
                        except Exception as e:
                            trans.rollback()
                            raise e
                except Exception as e:
                    st.error(f"Error adding project: {str(e)}")
    
    st.subheader("All Projects")
    
    try:
        # Get total count of projects
        with engine.connect() as conn:
            count_result = conn.execute(text("SELECT COUNT(*) FROM project")).scalar()
            total_projects = count_result
            st.write(f"Total projects in database: {total_projects}")
            
            if total_projects == 0:
                st.warning("No projects found in the database.")
                return
        
        # Fetch projects with their managers and allocation counts
        query = """
        SELECT 
            p.project_id,
            p.project_name,
            p.client_name,
            e.employee_name as manager_name,
            p.status,
            p.start_date,
            p.end_date,
            COALESCE(active.count, 0) as active_resources,
            COALESCE(inactive.count, 0) as past_resources
        FROM project p
        LEFT JOIN employee e ON p.manager_id = e.employee_code
        LEFT JOIN (
            SELECT project_id, COUNT(DISTINCT employee_code) as count
            FROM project_allocation
            WHERE status = 'Active'
            AND effective_from <= CURRENT_DATE
            AND (effective_to IS NULL OR effective_to >= CURRENT_DATE)
            GROUP BY project_id
        ) active ON active.project_id = p.project_id
        LEFT JOIN (
            SELECT project_id, COUNT(DISTINCT employee_code) as count
            FROM project_allocation
            WHERE status = 'Active'
            AND effective_from <= CURRENT_DATE
            AND effective_to < CURRENT_DATE
            GROUP BY project_id
        ) inactive ON inactive.project_id = p.project_id
        ORDER BY 
            CASE p.status 
                WHEN 'Active' THEN 1
                WHEN 'Inactive' THEN 2
                WHEN 'Completed' THEN 3
            END,
            p.project_name
        """
        
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
            
            # Configure and display the dataframe
            st.dataframe(
                df,
                hide_index=True,
                column_config={
                    "project_id": "Project ID",
                    "project_name": "Project Name",
                    "client_name": "Client",
                    "manager_name": "Project Manager",
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        help="Project status",
                        options=["Active", "Inactive", "Completed"],
                        required=True
                    ),
                    "start_date": "Start Date",
                    "end_date": "End Date",
                    "active_resources": st.column_config.NumberColumn(
                        "Active Resources",
                        help="Number of currently active team members"
                    ),
                    "past_resources": st.column_config.NumberColumn(
                        "Past Resources",
                        help="Number of previously allocated team members"
                    )
                },
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"Error loading projects: {str(e)}")
        logger.log_event(
            event_type="PROJECT_LIST_ERROR",
            description="Failed to fetch projects list",
            user=st.session_state.get('user', 'system'),
            details={"error": str(e)}
        )

def render_edit_project(engine, logger):
    """Edit project details"""
    st.header("Edit Project")
    
    # Get list of projects
    projects_df = pd.read_sql(
        "SELECT project_id, project_name, status FROM project ORDER BY project_name", 
        engine
    )
    
    # Project selection
    project_id = st.selectbox(
        "Select Project",
        projects_df['project_id'].tolist(),
        format_func=lambda x: f"{x} - {projects_df[projects_df['project_id'] == x]['project_name'].iloc[0]} ({projects_df[projects_df['project_id'] == x]['status'].iloc[0]})"
    )
    
    if project_id:
        # Get project details
        query = """
        SELECT 
            p.project_id,
            p.project_name,
            p.client_name,
            p.status,
            p.start_date,
            p.end_date,
            p.manager_id,
            COALESCE(e.employee_name, 'N/A') as manager_name 
        FROM project p 
        LEFT JOIN employee e ON p.manager_id = e.employee_code 
        WHERE p.project_id = :project_id
        """
        project_df = pd.read_sql(text(query), engine, params={'project_id': project_id})
        
        if not project_df.empty:
            project = project_df.iloc[0]
            
            # Get list of potential managers
            managers_df = pd.read_sql(
                "SELECT employee_code, employee_name FROM employee WHERE status = 'Active' ORDER BY employee_name",
                engine
            )
            
            # Edit form
            with st.form("edit_project_form"):
                new_name = st.text_input("Project Name", project['project_name'])
                new_client = st.text_input("Client Name", project['client_name'] or "")
                
                # Handle manager selection
                current_manager_id = project['manager_id'] if pd.notnull(project['manager_id']) else None
                manager_options = managers_df['employee_code'].tolist()
                default_index = next((i for i, x in enumerate(manager_options) if x == current_manager_id), 0) if current_manager_id else 0
                
                new_manager = st.selectbox(
                    "Project Manager",
                    manager_options,
                    format_func=lambda x: managers_df[managers_df['employee_code'] == x]['employee_name'].iloc[0],
                    index=default_index
                )
                new_status = st.selectbox("Status", ['Active', 'Inactive', 'Completed'], 
                                        index=['Active', 'Inactive', 'Completed'].index(project['status']))
                new_start_date = st.date_input("Start Date", project['start_date'])
                new_end_date = st.date_input("End Date", project['end_date'] if project['end_date'] else None)
                
                if st.form_submit_button("Update Project"):
                    try:
                        with engine.connect() as conn:
                            # Start transaction
                            trans = conn.begin()
                            try:
                                # Update project details
                                conn.execute(text("""
                                    UPDATE project 
                                    SET project_name = :project_name,
                                        client_name = :client_name,
                                        manager_id = :manager_id,
                                        status = :status,
                                        start_date = :start_date,
                                        end_date = :end_date
                                    WHERE project_id = :project_id
                                """), {
                                    "project_id": project_id,
                                    "project_name": new_name,
                                    "client_name": new_client,
                                    "manager_id": new_manager,
                                    "status": new_status,
                                    "start_date": new_start_date,
                                    "end_date": new_end_date
                                })
                                
                                # If project is marked as Completed, deactivate all active allocations
                                if new_status == 'Completed':
                                    conn.execute(text("""
                                        UPDATE project_allocation
                                        SET status = 'Inactive',
                                            effective_to = CURRENT_DATE
                                        WHERE project_id = :project_id
                                        AND status = 'Active'
                                    """), {"project_id": project_id})
                                
                                # Commit the transaction
                                trans.commit()
                                
                                logger.log_event(
                                    event_type="PROJECT_UPDATE",
                                    description=f"Updated project {project_id}",
                                    user=st.session_state.get('user', 'system'),
                                    details={
                                        'project_id': project_id,
                                        'new_name': new_name,
                                        'new_status': new_status,
                                        'new_manager': new_manager
                                    }
                                )
                                
                                st.success("Project updated successfully!")
                                st.rerun()
                            except Exception as e:
                                trans.rollback()
                                raise e
                    except Exception as e:
                        st.error(f"Error updating project: {str(e)}")

def render_manage_allocations(engine, logger):
    """Manage project allocations"""
    st.header("Manage Project Allocations")
    
    # Get list of projects
    projects_df = pd.read_sql(
        "SELECT project_id, project_name, status FROM project WHERE status != 'Completed' ORDER BY project_name", 
        engine
    )
    
    # Project selection
    project_id = st.selectbox(
        "Select Project",
        projects_df['project_id'].tolist(),
        format_func=lambda x: f"{x} - {projects_df[projects_df['project_id'] == x]['project_name'].iloc[0]}",
        key="allocation_project"
    )
    
    if project_id:
        # Show current allocations
        st.subheader("Current Team Members")
        query = """
        SELECT 
            pa.allocation_id,
            e.employee_code,
            e.employee_name,
            d.department_name,
            ewp.role,
            pa.allocation_percentage,
            pa.effective_from,
            pa.effective_to,
            pa.status
        FROM project_allocation pa
        JOIN employee e ON pa.employee_code = e.employee_code
        LEFT JOIN department d ON e.department_id = d.department_id
        LEFT JOIN employee_work_profile ewp ON e.employee_code = ewp.employee_code
        WHERE pa.project_id = :project_id
        ORDER BY pa.status DESC, e.employee_name
        """
        allocations_df = pd.read_sql(text(query), engine, params={'project_id': project_id})
        
        if not allocations_df.empty:
            st.dataframe(
                allocations_df[['employee_name', 'department_name', 'role', 'allocation_percentage', 'effective_from', 'effective_to', 'status']],
                hide_index=True,
                column_config={
                    "employee_name": "Employee Name",
                    "department_name": "Department",
                    "role": "Role (from profile)",
                    "allocation_percentage": "Allocation %",
                    "effective_from": "Start Date",
                    "effective_to": "End Date",
                    "status": "Status"
                }
            )
            
            # Edit allocation form
            st.subheader("Edit Allocation")
            # Create unique form key for edit allocation
            edit_form_id = "edit_allocation_form"
            with st.form(edit_form_id):
                st.subheader("Edit Allocation")
                allocation_id = st.selectbox(
                    "Select Member to Edit",
                    allocations_df['allocation_id'].tolist(),
                    format_func=lambda x: f"{allocations_df[allocations_df['allocation_id'] == x]['employee_name'].iloc[0]} ({allocations_df[allocations_df['allocation_id'] == x]['status'].iloc[0]})",
                    key="allocation_select"
                )
                
                if allocation_id:
                    current = allocations_df[allocations_df['allocation_id'] == allocation_id].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        new_percentage = st.number_input(
                            "Allocation Percentage", 
                            min_value=0.0, 
                            max_value=100.0, 
                            value=float(current['allocation_percentage']),
                            key="edit_percentage"
                        )
                        st.info(f"Role (from profile): {current['role'] if pd.notna(current['role']) else 'Not specified'}")
                        new_status = st.selectbox(
                            "Status", 
                            ['Active', 'Inactive'],
                            index=['Active', 'Inactive'].index(current['status']),
                            key="edit_status"
                        )
                    
                    with col2:
                        current_from = pd.to_datetime(current['effective_from']).date() if pd.notna(current['effective_from']) else date.today()
                        current_to = pd.to_datetime(current['effective_to']).date() if pd.notna(current['effective_to']) else None
                        
                        new_effective_from = st.date_input(
                            "Effective From",
                            value=current_from,
                            min_value=current_from,
                            key="edit_from_date"
                        )
                        new_effective_to = st.date_input(
                            "Effective To (Optional)",
                            value=current_to,
                            min_value=new_effective_from,
                            key="edit_to_date"
                        )
                    
                    change_reason = st.text_area("Reason for Change")
                    
                    if st.form_submit_button("Update Allocation"):
                        if not change_reason:
                            st.error("Please provide a reason for the change")
                        else:
                            try:
                                success = update_allocation(
                                    engine,
                                    allocation_id,
                                    current['employee_code'],
                                    new_percentage,
                                    new_status,
                                    new_effective_from,
                                    new_effective_to,
                                    change_reason,
                                    logger
                                )
                                if success:
                                    # Log the activity
                                    logger.log_event(
                                        event_type="ALLOCATION_UPDATE",
                                        description=f"Updated allocation for {current['employee_code']} on project {project_id}",
                                        user=st.session_state.get('user', 'system'),
                                        details={
                                            'allocation_id': allocation_id,
                                            'project_id': project_id,
                                            'employee_code': current['employee_code'],                                        'old_percentage': float(current['allocation_percentage']),
                                        'new_percentage': float(new_percentage),
                                        'old_status': current['status'],
                                        'new_status': new_status,
                                        'change_reason': change_reason
                                        }
                                    )
                                    st.success("Allocation updated successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to update allocation")
                            except Exception as e:
                                st.error(f"Error updating allocation: {str(e)}")

        # Add new allocation section
        st.subheader("Add New Team Member")
        with st.form("add_resource_form"):
            # Get available employees (not currently allocated to this project)
            available_employees_query = """
            SELECT e.employee_code, e.employee_name, d.department_name
            FROM employee e
            LEFT JOIN department d ON e.department_id = d.department_id
            WHERE e.status = 'Active'
            AND e.employee_code NOT IN (
                SELECT employee_code
                FROM project_allocation
                WHERE project_id = :project_id AND status = 'Active'
            )
            ORDER BY e.employee_name
            """
            available_employees_df = pd.read_sql(text(available_employees_query), engine, params={'project_id': project_id})
            
            if not available_employees_df.empty:
                selected_employee = st.selectbox(
                    "Select Employee",
                    options=available_employees_df['employee_code'].tolist(),
                    format_func=lambda x: f"{available_employees_df[available_employees_df['employee_code']==x]['employee_name'].iloc[0]} ({available_employees_df[available_employees_df['employee_code']==x]['department_name'].iloc[0]})"
                )
                
                allocation_percentage = st.number_input("Allocation Percentage", min_value=0, max_value=100, value=100)
                st.info("Role information is managed in the employee's work profile")
                effective_from = st.date_input("Effective From", min_value=date.today())
                effective_to = st.date_input("Effective To", min_value=effective_from)
                
                submit_clicked = st.form_submit_button("Add Team Member")
                
                if submit_clicked:
                    try:
                        with engine.connect() as conn:
                            # Start transaction
                            trans = conn.begin()
                            try:
                                # Get a valid created_by employee code
                                created_by = get_valid_created_by(engine)
                                if not created_by:
                                    st.error("No valid employee found for created_by field")
                                    return
                                
                                # Insert new allocation
                                conn.execute(text("""
                                    INSERT INTO project_allocation 
                                    (employee_code, project_id, allocation_percentage, effective_from, 
                                     effective_to, status, created_by, change_reason, created_at)
                                    VALUES (:employee_code, :project_id, :allocation_percentage, :effective_from, 
                                            :effective_to, :status, :created_by, :change_reason, :created_at)
                                """), {
                                    "employee_code": selected_employee,
                                    "project_id": project_id,
                                    "allocation_percentage": allocation_percentage,
                                    "effective_from": effective_from,
                                    "effective_to": effective_to,
                                    "status": "Active",
                                    "created_by": created_by,
                                    "change_reason": "Initial allocation",
                                    "created_at": datetime.now()
                                })
                                
                                # Commit the transaction
                                trans.commit()
                                
                                logger.log_event(
                                    event_type="ALLOCATION_CREATE",
                                    description=f"Added new team member to project {project_id}",
                                    user=st.session_state.get('user', created_by),
                                    details={
                                        'project_id': project_id,
                                        'employee_code': selected_employee,
                                        'allocation_percentage': allocation_percentage
                                    }
                                )
                                
                                st.success("Team member added successfully!")
                                st.rerun()
                                
                            except Exception as e:
                                trans.rollback()
                                raise e
                                
                    except Exception as e:
                        st.error(f"Error adding team member: {str(e)}")
            else:
                st.info("No available employees to add to this project.")
                # Add a disabled submit button to satisfy Streamlit's form requirements
                st.form_submit_button("Add Team Member", disabled=True)