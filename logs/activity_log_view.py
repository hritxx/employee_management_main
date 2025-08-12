"""
Activity Logs View Module for HR Management System

This module provides the Streamlit UI components for displaying system activity logs.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from logs.activity_logger import get_logger

def render_activity_logs(engine=None):
    """
    Render the activity logs page in Streamlit
    
    Args:
        engine: SQLAlchemy engine (optional)
    """
    # Get logger instance
    logger = get_logger(engine)
    
    st.subheader("System Activity Logs")
    
    # Create tabs for different log views
    log_tabs = st.tabs(["All Logs", "User Activity", "File Operations", "Allocations", "Maintenance"])
    
    with log_tabs[0]:
        render_all_logs(logger)
    
    with log_tabs[1]:
        render_user_logs(logger)
    
    with log_tabs[2]:
        render_file_logs(logger)
        
    with log_tabs[3]:
        render_allocation_logs(logger)
        
    with log_tabs[4]:
        render_maintenance_view(logger)

def render_all_logs(logger, limit=100):
    """Render all system logs with filtering options"""
    # Get logs (all types)
    logs = logger.get_logs(limit=limit)
    
    # Add filtering options
    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.slider("Days to show", 1, 30, 7)
    
    with col2:
        event_types = sorted(logs['event_type'].unique().tolist()) if not logs.empty else []
        selected_types = st.multiselect("Event types", ["All"] + event_types, default="All")
    
    with col3:
        search_term = st.text_input("Search logs", placeholder="Enter keywords...")
    
    # Apply filters
    if not logs.empty:
        # Filter by days
        logs['days_ago'] = (datetime.now() - logs['timestamp']).dt.days
        filtered_logs = logs[logs['days_ago'] <= days]
        
        # Filter by event type
        if selected_types and "All" not in selected_types:
            filtered_logs = filtered_logs[filtered_logs['event_type'].isin(selected_types)]
        
        # Filter by search term
        if search_term:
            search_term = search_term.lower()
            filtered_logs = filtered_logs[
                filtered_logs['description'].str.lower().str.contains(search_term) | 
                filtered_logs['user'].str.lower().str.contains(search_term) | 
                filtered_logs['details'].astype(str).str.lower().str.contains(search_term)
            ]
        
        # Process logs for display
        display_logs = process_logs_for_display(filtered_logs)
        
        if not display_logs.empty:
            # Display the table
            st.dataframe(display_logs, use_container_width=True)
            
            # Show log stats
            st.info(f"Showing {len(filtered_logs)} of {len(logs)} total logs")
        else:
            st.info("No logs match the selected filters")
    else:
        st.info("No logs found")

def render_user_logs(logger):
    """Render user activity logs (logins, logouts, etc.)"""
    logs = logger.get_logs(limit=200)
    
    if not logs.empty:
        # Filter for user activity
        user_event_types = ['USER_LOGIN', 'USER_LOGOUT']
        user_logs = logs[logs['event_type'].isin(user_event_types)]
        
        if not user_logs.empty:
            # Process logs for display
            display_logs = process_logs_for_display(user_logs)
            
            # Display the table
            st.dataframe(display_logs, use_container_width=True)
            
            # Show login/logout summary by user
            try:
                # Extract user details from logs
                user_stats = []
                for _, log in user_logs.iterrows():
                    try:
                        if 'details' in log and log['details']:
                            details = json.loads(log['details']) if isinstance(log['details'], str) else log['details']
                            if 'full_name' in details:
                                user_name = details['full_name']
                                event_type = log['event_type']
                                timestamp = log['timestamp']
                                
                                user_stats.append({
                                    'User': user_name,
                                    'Event': event_type,
                                    'Time': timestamp
                                })
                    except:
                        pass
                
                if user_stats:
                    st.subheader("Recent User Activity")
                    user_df = pd.DataFrame(user_stats)
                    user_df = user_df.sort_values('Time', ascending=False)
                    st.dataframe(user_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing user statistics: {str(e)}")
        else:
            st.info("No user activity logs found")
    else:
        st.info("No logs found")

def render_file_logs(logger):
    """Render file upload and processing logs"""
    logs = logger.get_logs(limit=200)
    
    if not logs.empty:
        # Filter for file operations
        file_event_types = ['FILE_UPLOAD', 'FILE_PROCESSING']
        file_logs = logs[logs['event_type'].isin(file_event_types)]
        
        if not file_logs.empty:
            # Process logs for display
            display_logs = process_logs_for_display(file_logs)
            
            # Display the table
            st.dataframe(display_logs, use_container_width=True)
        else:
            st.info("No file operation logs found")
    else:
        st.info("No logs found")

def render_allocation_logs(logger):
    """Render allocation change logs"""
    logs = logger.get_logs(limit=200)
    
    if not logs.empty:
        # Filter for allocation operations
        allocation_event_types = ['ALLOCATION_CHANGE', 'ALLOCATION_UPDATE', 'PROJECT_CHANGE']
        allocation_logs = logs[logs['event_type'].str.contains('ALLOCATION|PROJECT', case=False, regex=True)]
        
        if not allocation_logs.empty:
            # Process logs for display
            display_logs = process_logs_for_display(allocation_logs)
            
            # Display the table
            st.dataframe(display_logs, use_container_width=True)
        else:
            st.info("No allocation logs found")
    else:
        st.info("No logs found")

def render_maintenance_view(logger):
    """Render maintenance view with log purging option"""
    st.subheader("Log Maintenance")
    
    # Show log stats
    try:
        stats = logger.get_log_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Logs", stats['total_count'])
        
        # Display log retention policy
        st.markdown("### Log Retention Policy")
        st.info("Logs are automatically purged after 30 days to maintain system performance.")
        
        # Manual purge option
        st.markdown("### Manual Log Purge")
        st.warning("⚠️ This action will permanently delete old logs and cannot be undone.")
        
        col1, col2 = st.columns(2)
        with col1:
            days_to_keep = st.slider("Days of logs to keep", 7, 90, 30)
        
        with col2:
            if st.button("Purge Old Logs"):
                with st.spinner("Purging old logs..."):
                    # Check if user has permission
                    if 'username' in st.session_state and st.session_state['username']:
                        # Perform the purge
                        deleted_count = logger.purge_old_logs(days_to_keep)
                        
                        # Log the manual purge
                        logger.log_event(
                            event_type="MANUAL_LOG_PURGE",
                            description=f"Manual log purge: Removed {deleted_count} records older than {days_to_keep} days",
                            user=st.session_state.get('username', 'hr_user'),
                            details={
                                "user_full_name": st.session_state.get('user_full_name', 'Unknown'),
                                "records_deleted": deleted_count,
                                "retention_days": days_to_keep,
                                "timestamp": str(datetime.now())
                            }
                        )
                        
                        st.success(f"Successfully purged {deleted_count} logs older than {days_to_keep} days")
                    else:
                        st.error("You must be logged in to perform this operation")
        
    except Exception as e:
        st.error(f"Error loading maintenance view: {str(e)}")

def process_logs_for_display(logs_df):
    """Process logs dataframe for display"""
    if logs_df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    display_df = logs_df.copy()
    
    # Format timestamp
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract user details
    display_df['user_name'] = 'Unknown'
    for idx, row in display_df.iterrows():
        if 'details' in row and row['details']:
            try:
                details = json.loads(row['details']) if isinstance(row['details'], str) else row['details']
                if 'user_full_name' in details:
                    display_df.at[idx, 'user_name'] = details['user_full_name']
            except:
                pass
    
    # Select and rename columns for display
    columns_to_display = ['log_id', 'event_type', 'description', 'user', 'user_name', 'timestamp']
    columns_to_display = [col for col in columns_to_display if col in display_df.columns]
    
    if columns_to_display:
        result_df = display_df[columns_to_display]
        column_names = {
            'log_id': 'ID', 
            'event_type': 'Event Type', 
            'description': 'Description', 
            'user': 'Username', 
            'user_name': 'Full Name',
            'timestamp': 'Timestamp'
        }
        result_df = result_df.rename(columns={k: v for k, v in column_names.items() if k in result_df.columns})
        return result_df
    
    return pd.DataFrame()