import streamlit as st
import subprocess
import os
import glob
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from logs.activity_logger import get_logger

def render_backup_page():
    """Render the backup management page"""
    st.header("Database Backup Management")
    
    # Get the global activity logger instance
    logger = get_logger()
    
    # Create tabs for different backup operations
    tab1, tab2, tab3 = st.tabs(["Manual Backup", "Backup History", "Settings"])
    
    with tab1:
        render_manual_backup(logger)
    
    with tab2:
        render_backup_history()
    
    with tab3:
        render_backup_settings()

def render_manual_backup(logger):
    """Render manual backup section"""
    st.header("Manual Database Backup")
    
    # Get user details for logging
    user_name = st.session_state.get('username', None)
    user_details = {}
    
    # If no username is set in session state, ask for name
    if not user_name:
        st.warning("Please provide your name for backup tracking purposes.")
        user_name = st.text_input("Your Name:", placeholder="Enter your full name", key="backup_user_name")
        if user_name:
            st.session_state['backup_user_name'] = user_name
            user_details['name'] = user_name
    else:
        user_details['username'] = user_name
        # If we have the username but not a display name, ask for it
        if 'backup_user_name' not in st.session_state:
            display_name = st.text_input("Your Name:", placeholder="Enter your full name", key="backup_user_name")
            if display_name:
                st.session_state['backup_user_name'] = display_name
                user_details['name'] = display_name
        else:
            user_details['name'] = st.session_state['backup_user_name']
    
    st.info("💡 **When to use manual backup:**")
    st.markdown("""
    - After adding new employees
    - After updating project allocations
    - Before making major changes
    - After importing bulk data
    - When you want to save current state
    """)
    
    # Backup reason selection
    backup_reasons = [
        "Manual backup after data update",
        "Before major changes",
        "After employee onboarding",
        "After project allocation changes",
        "After bulk data import",
        "Regular manual backup",
        "Custom reason"
    ]
    
    selected_reason = st.selectbox("Backup Reason:", backup_reasons)
    
    if selected_reason == "Custom reason":
        custom_reason = st.text_input("Enter custom reason:", placeholder="Describe why you're creating this backup")
        backup_reason = custom_reason if custom_reason.strip() else "Manual backup"
    else:
        backup_reason = selected_reason
    
    # Backup button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔄 Create Backup Now", type="primary", use_container_width=True):
            if backup_reason.strip():
                create_manual_backup(backup_reason, logger)
            else:
                st.error("Please provide a reason for the backup")

def create_manual_backup(reason, logger):
    """Create a manual backup with progress indication"""
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Initializing backup...")
        progress_bar.progress(10)
        
        # Create backup using direct PostgreSQL connection
        import sqlalchemy
        from sqlalchemy import create_engine, text
        
        status_text.text("🔄 Connecting to database...")
        progress_bar.progress(20)
        
        # Create backup directory
        backup_dir = "/app/data/backups"
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{backup_dir}/employee_db_backup_{timestamp}.sql"
        
        status_text.text("🔄 Creating database dump...")
        progress_bar.progress(40)
        
        # Use pg_dump directly within the same network
        # Since we're in the same Docker network, we can connect directly to the database
        cmd = [
            "pg_dump", 
            "-h", "postgres",  # Use service name from docker-compose
            "-U", "postgres", 
            "-d", "employee_db",
            "--clean", "--if-exists",
            "-f", backup_file
        ]
        
        # Set environment variable for password (no password prompt)
        env = os.environ.copy()
        env['PGPASSWORD'] = 'postgres123'  # From docker-compose
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            status_text.text("🔄 Compressing backup...")
            progress_bar.progress(70)
            
            # Compress backup
            compress_cmd = ["gzip", backup_file]
            subprocess.run(compress_cmd, check=True)
            compressed_file = f"{backup_file}.gz"
            
            status_text.text("🔄 Verifying backup...")
            progress_bar.progress(90)
            
            # Verify backup
            if verify_backup_file(compressed_file):
                progress_bar.progress(100)
                status_text.text("✅ Backup completed successfully!")
                
                # Log the manual backup
                logger.log_event(
                    event_type="MANUAL_BACKUP",
                    description=f"Manual backup created by {st.session_state.get('backup_user_name', 'user')}",
                    user=st.session_state.get('username', 'hr_user'),
                    details={
                        "backup_file": compressed_file,
                        "reason": reason,
                        "initiated_by": "UI",
                        "user_name": st.session_state.get('backup_user_name', 'Unknown'),
                        "file_size": os.path.getsize(compressed_file) if os.path.exists(compressed_file) else 0
                    }
                )
                
                st.success(f"✅ **Backup Created Successfully!**")
                st.info(f"👤 **User:** {st.session_state.get('backup_user_name', 'Unknown')}")
                st.info(f"📁 **File:** `{os.path.basename(compressed_file)}`")
                st.info(f"📋 **Reason:** {reason}")
                
                # Show file size
                if os.path.exists(compressed_file):
                    file_size = os.path.getsize(compressed_file) / (1024 * 1024)
                    st.info(f"📊 **Size:** {file_size:.2f} MB")
                
            else:
                progress_bar.progress(100)
                status_text.text("❌ Backup verification failed")
                st.error("❌ **Backup verification failed!** Please check the logs and try again.")
                
                logger.log_event(
                    event_type="MANUAL_BACKUP_ERROR",
                    description="Manual backup verification failed",
                    user=st.session_state.get('username', 'hr_user'),
                    details={
                        "reason": reason, 
                        "error": "Verification failed",
                        "user_name": st.session_state.get('backup_user_name', 'Unknown')
                    }
                )
        else:
            progress_bar.progress(100)
            status_text.text("❌ Backup creation failed")
            st.error("❌ **Backup creation failed!** Please check the database connection and try again.")
            st.error(f"Error details: {result.stderr}")
            
            logger.log_event(
                event_type="MANUAL_BACKUP_ERROR",
                description="Manual backup creation failed",
                user=st.session_state.get('username', 'hr_user'),
                details={
                    "reason": reason, 
                    "error": result.stderr,
                    "user_name": st.session_state.get('backup_user_name', 'Unknown')
                }
            )
            
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("❌ Backup failed")
        st.error(f"❌ **Backup failed:** {str(e)}")
        
        logger.log_event(
            event_type="MANUAL_BACKUP_ERROR",
            description=f"Manual backup failed with error: {str(e)}",
            user=st.session_state.get('username', 'hr_user'),
            details={
                "reason": reason, 
                "error": str(e),
                "user_name": st.session_state.get('backup_user_name', 'Unknown')
            }
        )

def verify_backup_file(backup_file):
    """Verify backup file integrity"""
    try:
        if not os.path.exists(backup_file):
            return False
        
        # Check if file is not empty
        file_size = os.path.getsize(backup_file)
        if file_size == 0:
            return False
        
        # Test gzip integrity
        test_cmd = ["gunzip", "-t", backup_file]
        result = subprocess.run(test_cmd, capture_output=True)
        
        return result.returncode == 0
        
    except Exception as e:
        return False

def render_backup_history():
    """Render backup history section"""
    st.header("Backup History")
    
    backup_dir = "/app/data/backups"
    
    if not os.path.exists(backup_dir):
        st.warning("No backup directory found.")
        return
    
    # Get all backup files
    backup_files = glob.glob(f"{backup_dir}/employee_db_backup_*.sql.gz")
    
    if not backup_files:
        st.info("No backup files found.")
        return
    
    # Sort by modification time (newest first)
    backup_files.sort(key=os.path.getmtime, reverse=True)
    
    # Try to get backup logs to match with files
    try:
        from logs.activity_logger import get_logger
        logger = get_logger()
        logs = logger.get_logs(event_type="MANUAL_BACKUP", limit=100)
        
        # Create a mapping from backup filename to user
        backup_to_user = {}
        for _, log in logs.iterrows():
            try:
                if 'details' in log and log['details']:
                    details = json.loads(log['details']) if isinstance(log['details'], str) else log['details']
                    if 'backup_file' in details:
                        backup_file = details['backup_file']
                        user_name = details.get('user_name', log.get('user', 'Unknown'))
                        backup_to_user[os.path.basename(backup_file)] = user_name
            except:
                pass
    except Exception as e:
        backup_to_user = {}
        st.warning(f"Could not fetch user information for backups: {str(e)}")
    
    # Create backup history table
    backup_data = []
    for backup_file in backup_files:
        file_name = os.path.basename(backup_file)
        file_size = os.path.getsize(backup_file) / (1024 * 1024)  # MB
        created_time = datetime.fromtimestamp(os.path.getmtime(backup_file))
        age_days = (datetime.now() - created_time).days
        
        # Try to get user who created this backup
        created_by = backup_to_user.get(file_name, "Unknown")
        
        backup_data.append({
            "File Name": file_name,
            "Created": created_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Created By": created_by,
            "Size (MB)": f"{file_size:.2f}",
            "Age (Days)": age_days,
            "Status": "✅ Valid" if age_days <= 30 else "⚠️ Old"
        })
    
    # Display as dataframe
    if backup_data:
        df = pd.DataFrame(backup_data)
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "File Name": st.column_config.TextColumn("File Name", width="large"),
                "Created": "Created Date",
                "Created By": "Created By",
                "Size (MB)": "Size (MB)",
                "Age (Days)": "Age (Days)",
                "Status": "Status"
            }
        )
        
        # Summary stats
        total_size = sum([os.path.getsize(f) for f in backup_files]) / (1024 * 1024)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Backups", len(backup_files))
        with col2:
            st.metric("Total Size", f"{total_size:.2f} MB")
        with col3:
            latest_backup = max(backup_files, key=os.path.getmtime)
            latest_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(latest_backup))).days
            st.metric("Latest Backup Age", f"{latest_age} days")

def render_backup_settings():
    """Render backup settings section"""
    st.header("Backup Settings")
    
    st.info("ℹ️ **Current Configuration:**")
    
    # Show current settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📁 Backup Location:**")
        st.code("/app/data/backups/")
        
        st.markdown("**🗄️ Database:**")
        st.code("employee_db")
        
        st.markdown("**⏰ Automatic Schedule:**")
        st.code("Every Sunday at 2:00 AM")
    
    with col2:
        st.markdown("**🗂️ Retention Period:**")
        st.code("30 days")
        
        st.markdown("**🔧 Backup Format:**")
        st.code("PostgreSQL dump (gzipped)")
        
        st.markdown("**📊 Compression:**")
        st.code("Enabled (gzip)")
    
    st.markdown("---")
    
    st.markdown("**🎯 Best Practices:**")
    st.markdown("""
    - Create manual backups before major updates
    - Weekly automatic backups are scheduled
    - Backups are compressed to save space
    - Old backups (30+ days) are automatically deleted
    - Always verify backup integrity after creation
    """)
    
    # Quick backup test
    st.markdown("**🧪 Test Backup System:**")
    if st.button("Test Backup Connection"):
        test_backup_system()

def test_backup_system():
    """Test if backup system is working"""
    try:
        # Test backup directory
        backup_dir = "/app/data/backups"
        if os.path.exists(backup_dir):
            st.success("✅ Backup directory exists")
        else:
            os.makedirs(backup_dir, exist_ok=True)
            st.success("✅ Backup directory created")
        
        # Test database connection
        try:
            # Test using pg_isready directly - should work in Docker container
            cmd = ["pg_isready", "-h", "postgres", "-U", "postgres"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("✅ Database is ready")
            else:
                st.error(f"❌ Database connection failed: {result.stderr}")
            
        except Exception as db_error:
            st.error(f"❌ Database connection test failed: {str(db_error)}")
        
        # Check for pg_dump availability
        try:
            pg_dump_version = subprocess.run(["pg_dump", "--version"], capture_output=True, text=True)
            if pg_dump_version.returncode == 0:
                st.success(f"✅ pg_dump is available: {pg_dump_version.stdout.strip()}")
            else:
                st.error("❌ pg_dump not found or not working")
        except Exception as pg_error:
            st.error(f"❌ pg_dump test failed: {str(pg_error)}")
        
        st.info("🎉 Backup system test completed!")
        
    except Exception as e:
        st.error(f"❌ Backup system test failed: {str(e)}")
