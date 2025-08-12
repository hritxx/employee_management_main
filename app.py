"""
Main Streamlit application for HR Management System
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import sys
from pages.summary_reports import render_summary_reports
from pages.query_assistant import render_ai_query_assistant
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from pages.tasks_summariser import task_summarizer
from pages.custom_queries import render_custom_queries
from pages.file_upload import render_file_upload
from pages.report import render_standard_reports
from logs.activity_log_view import render_activity_logs
from logs.activity_logger import get_logger
from pages.allocations import render_allocations
from pages.backup import render_backup_page
from auth.auth import AuthManager

# Load environment variables
load_dotenv()

# Initialize auth manager
auth_manager = AuthManager()

# URL encode the password if it contains special characters
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Add database connection details for SQLAlchemy engine
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Create SQLAlchemy engine for PostgreSQL
encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Initialize activity logger with the main engine
activity_logger = get_logger(engine)

# Import configurations
from config.config import app_config, etl_config, db_config

# Set page config first with custom theme and favicon
st.set_page_config(
    page_title="Aganitha HR Management System",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for blue theme and professional look
st.markdown(
    """
    <style>
    /* Main background and font */
    .stApp { background-color: #f4f8fb; font-family: 'Segoe UI', Arial, sans-serif; }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1f4e79 80%, #4472c4 100%);
        color: #fff;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1v0mbdj p, .st-emotion-cache-1v0mbdj h1, .st-emotion-cache-1v0mbdj h2, .st-emotion-cache-1v0mbdj h3 {
        color: #1f4e79 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: #eaf1fb;
        color: #1f4e79;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1f4e79;
        color: #fff !important;
    }
    .stButton>button {
        background-color: #1f4e79;
        color: #fff;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        padding: 0.5em 1.2em;
    }
    .stButton>button:hover {
        background-color: #4472c4;
        color: #fff;
    }
    .stMetric {
        background: #eaf1fb;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1f4e79 !important;
    }
    .st-bb, .st-cq, .st-cp, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
        color: #1f4e79 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with company branding
with st.sidebar:
    st.image("https://www.aganitha.ai/wp-content/uploads/2023/05/aganitha-logo.png", width=180)
    st.markdown("""
    <h2 style='color:#fff; margin-bottom:0;'>Aganitha HR Portal</h2>
    <p style='color:#eaf1fb; font-size:1.1em;'>Empowering People, Enabling Growth</p>
    <hr style='border:1px solid #eaf1fb; margin:1em 0;'>
    """, unsafe_allow_html=True)
    st.markdown(f"<b>User:</b> {auth_manager.get_current_user()}", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
    st.markdown("<br><br><small style='color:#eaf1fb;'>© 2025 Aganitha Cognitive Solutions</small>", unsafe_allow_html=True)

# Configure logging
logger = logging.getLogger(__name__)

from core.database import db_pool
from core.etl import ETLPipeline
from core.models import create_tables

# Add the get_available_tables function
def get_available_tables():
    """Get list of tables in the database"""
    try:
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        result = pd.read_sql(query, engine)
        return result['table_name'].tolist()
    except Exception as e:
        st.error(f"Error getting tables: {e}")
        return []

def initialize_database():
    """Initialize database tables"""
    try:
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        st.error("Failed to initialize database tables. Please check the logs.")

def render_authenticated_app():
    """Render the main application after authentication"""
    # Initialize database tables
    initialize_database()

    # Debug logging for database configuration
    logger.info("Database Configuration:")
    logger.info(f"Host: {db_config.host}")
    logger.info(f"Database: {db_config.database}")
    logger.info(f"User: {db_config.user}")
    logger.info(f"Port: {db_config.port}")

    # Page title and welcome
    st.markdown("""
    <div style='display:flex; align-items:center; gap:1em;'>
        <img src='https://www.aganitha.ai/wp-content/uploads/2023/05/aganitha-logo.png' width='60'>
        <h1 style='color:#1f4e79; margin-bottom:0;'>Aganitha Employee Management System</h1>
    </div>
    <p style='color:#4472c4; font-size:1.2em; margin-top:0;'></p>
    <hr style='border:1px solid #1f4e79;'>
    """, unsafe_allow_html=True)

    # Create tabs
    tabs = st.tabs([
        "📂 File Upload", 
        "📊 Predefined Reports", 
        "🔍 Custom Queries", 
        "🤖 AI Query Assistant",
        "🤖 Tasks Summariser",
        "📋 Standard Reports",
        "🎯 Allocations",
        "⚙️ Settings"
    ])
    
    with tabs[0]:
        render_file_upload(db_pool)

    with tabs[1]:
        render_summary_reports()
    
    with tabs[2]:
        render_custom_queries(engine)
    
    with tabs[3]:
        render_ai_query_assistant(engine, get_available_tables, GEMINI_API_KEY)

    with tabs[4]:
        task_summarizer()   

    with tabs[5]:
        render_standard_reports(engine, db_pool)

    with tabs[6]:
        render_allocations(engine)

    #with tabs[7]:
    #    render_activity_logs(engine)

    # --- Settings Tab ---
    with tabs[7]:
        settings_tabs = st.tabs(["Logs", "Backup", "Clear DB"])
        with settings_tabs[0]:
            st.subheader("Application Logs")
            render_activity_logs(engine)
        with settings_tabs[1]:
            render_backup_page()
        with settings_tabs[2]:
            st.subheader("🗑️ Clear Database Tables")
            
            st.warning("""
            ⚠️ **WARNING: This action is irreversible!**
            
            This will delete all data from the following tables while preserving table structure:
            - Employees, Departments, Designations
            - Projects and Allocations
            - Attendance and Timesheet data
            - Financial and Personal information
            - Task summaries and upload logs
            """)
            
            # Add confirmation checkbox
            confirm_clear = st.checkbox("I understand this will permanently delete all data", key="confirm_clear")
            
            if confirm_clear:
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🗑️ Clear All Data", key="clear_db_btn", type="primary"):
                        with st.spinner("Clearing database tables..."):
                            try:
                                from clear_db import clear_tables
                                success, message = clear_tables()
                                
                                if success:
                                    st.success(f"✅ {message}")
                                    activity_logger.log_event("DATABASE", f"Clear All Tables - SUCCESS: {message}", auth_manager.get_current_user())
                                    st.balloons()
                                else:
                                    st.error(f"❌ {message}")
                                    activity_logger.log_event("DATABASE", f"Clear All Tables - ERROR: {message}", auth_manager.get_current_user())
                                    
                            except Exception as e:
                                error_msg = f"Unexpected error: {str(e)}"
                                st.error(f"❌ {error_msg}")
                                activity_logger.log_event("DATABASE", f"Clear All Tables - ERROR: {error_msg}", auth_manager.get_current_user())
                with col2:
                    st.info("💡 Consider creating a backup before clearing data using the Backup tab.")
            else:
                st.info("Please confirm that you understand the consequences before proceeding.")

def main():
    """Main application entry point"""
    # Check authentication
    if not auth_manager.is_authenticated():
        if not auth_manager.login_form():
            return
    
    # If authenticated, render the main app
    render_authenticated_app()

if __name__ == "__main__":
    main()