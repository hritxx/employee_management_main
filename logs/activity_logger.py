"""
Activity Logger Module for HR Management System

This module provides functionality to log system activities and display them in the UI.
It uses SQLAlchemy for database interactions and follows enterprise-level design patterns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, 
    Text, DateTime, ForeignKey, select, desc, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Create SQLAlchemy Base
Base = declarative_base()

# Define the SystemLog model
class SystemLog(Base):
    """System activity log model"""
    __tablename__ = 'system_logs'
    
    log_id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False, index=True)
    user = Column(String(100), nullable=True)
    description = Column(Text, nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)
    
    def __repr__(self):
        return f"<SystemLog(log_id={self.log_id}, event_type='{self.event_type}', timestamp='{self.timestamp}')>"


class ActivityLogger:
    """
    Activity Logger class for tracking system activities
    
    This class provides methods to log various system activities and retrieve logs
    for display in the UI. It uses SQLAlchemy for database interactions.
    """
    
    def __init__(self, engine=None):
        """
        Initialize the ActivityLogger with a SQLAlchemy engine
        
        Args:
            engine: SQLAlchemy engine (optional, will create from env vars if not provided)
        """
        try:
            if engine is None:
                # Create SQLAlchemy engine from environment variables
                db_host = os.getenv('DB_HOST', 'localhost')
                db_port = os.getenv('DB_PORT', '5432')
                db_name = os.getenv('DB_NAME', 'employee_management_db')
                db_user = os.getenv('DB_USER', 'hr_admin')
                db_password = os.getenv('DB_PASSWORD', '')
                
                # URL encode the password if it contains special characters
                encoded_password = quote_plus(db_password)
                
                # Create SQLAlchemy engine for PostgreSQL with connection pooling
                db_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
                self.engine = create_engine(db_url, pool_size=5, max_overflow=10)
            else:
                self.engine = engine
                
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Ensure the logs table exists
            self.create_logs_table()
        except Exception as e:
            logger.error(f"Failed to initialize ActivityLogger: {e}")
            # Create a dummy engine and session for fallback
            self.engine = None
            self.Session = None
        
    def create_logs_table(self):
        """Create the system_logs table if it doesn't exist"""
        try:
            Base.metadata.create_all(self.engine, tables=[SystemLog.__table__])
            logger.info("System logs table created or already exists")
        except Exception as e:
            logger.error(f"Error creating system logs table: {e}")
    
    def log_event(self, event_type: str, description: str, user: Optional[str] = None, 
                  details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a system event
        
        Args:
            event_type: Type of event (e.g., 'FILE_UPLOAD', 'QUERY', 'LOGIN')
            description: Brief description of the event
            user: Username or identifier of the user who triggered the event
            details: Additional details about the event as a dictionary
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        try:
            if self.Session is None:
                logger.error("Cannot log event: Session is not initialized")
                return False
                
            # Convert details dictionary to string if provided
            details_str = None
            if details:
                import json
                details_str = json.dumps(details)
            
            # Create a new log entry
            log_entry = SystemLog(
                event_type=event_type,
                user=user,
                description=description,
                details=details_str,
                timestamp=datetime.now()
            )
            
            # Add to database
            with self.Session() as session:
                session.add(log_entry)
                session.commit()
                
            logger.info(f"Logged event: {event_type} - {description}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return False
    
    def log_file_upload(self, filename: str, file_type: str, user: Optional[str] = None, 
                        status: str = "SUCCESS", details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a file upload event
        
        Args:
            filename: Name of the uploaded file
            file_type: Type of file uploaded
            user: Username of the uploader
            status: Status of the upload (SUCCESS, FAILED)
            details: Additional details about the upload
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        description = f"File upload: {filename} ({file_type}) - {status}"
        return self.log_event("FILE_UPLOAD", description, user, details)
    
    def log_file_processing(self, filename: str, records_processed: int, 
                           records_success: int, records_failed: int,
                           user: Optional[str] = None) -> bool:
        """
        Log a file processing event
        
        Args:
            filename: Name of the processed file
            records_processed: Number of records processed
            records_success: Number of successfully processed records
            records_failed: Number of failed records
            user: Username who initiated the processing
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        description = f"File processed: {filename}"
        details = {
            "records_processed": records_processed,
            "records_success": records_success,
            "records_failed": records_failed
        }
        return self.log_event("FILE_PROCESSING", description, user, details)
    
    def log_query(self, query_text: str, user: Optional[str] = None, 
                 query_type: str = "CUSTOM", status: str = "SUCCESS") -> bool:
        """
        Log a database query event
        
        Args:
            query_text: The query that was executed
            user: Username who executed the query
            query_type: Type of query (CUSTOM, AI_GENERATED)
            status: Status of the query execution
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        # Truncate query text if it's too long for the log
        if len(query_text) > 500:
            query_text_short = query_text[:500] + "..."
        else:
            query_text_short = query_text
            
        description = f"{query_type} query executed: {query_text_short}"
        details = {
            "query": query_text,
            "status": status
        }
        return self.log_event("QUERY", description, user, details)
    
    def log_ai_query(self, user_query: str, generated_sql: str, 
                    user: Optional[str] = None, status: str = "SUCCESS") -> bool:
        """
        Log an AI-generated query event
        
        Args:
            user_query: Natural language query from the user
            generated_sql: SQL query generated by AI
            user: Username who initiated the query
            status: Status of the query generation and execution
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        description = f"AI Query: {user_query[:100]}..."
        details = {
            "user_query": user_query,
            "generated_sql": generated_sql,
            "status": status
        }
        return self.log_event("AI_QUERY", description, user, details)
    
    def get_logs(self, event_type: Optional[str] = None, 
                limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """
        Get system logs as a pandas DataFrame
        
        Args:
            event_type: Filter logs by event type (optional)
            limit: Maximum number of logs to retrieve
            offset: Offset for pagination
            
        Returns:
            pd.DataFrame: DataFrame containing the logs
        """
        try:
            if self.Session is None:
                logger.error("Cannot get logs: Session is not initialized")
                return pd.DataFrame()
                
            with self.Session() as session:
                # Build query
                query = select(SystemLog).order_by(desc(SystemLog.timestamp))
                
                # Apply event type filter if provided
                if event_type:
                    query = query.where(SystemLog.event_type == event_type)
                
                # Apply limit and offset
                query = query.limit(limit).offset(offset)
                
                # Execute query
                result = session.execute(query).scalars().all()
                
                # Convert to list of dictionaries
                logs_data = []
                for log in result:
                    log_dict = {
                        "log_id": log.log_id,
                        "event_type": log.event_type,
                        "user": log.user,
                        "description": log.description,
                        "details": log.details,
                        "timestamp": log.timestamp
                    }
                    logs_data.append(log_dict)
                
                # Return as DataFrame
                return pd.DataFrame(logs_data)
                
        except Exception as e:
            logger.error(f"Error retrieving logs: {e}")
            return pd.DataFrame()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get statistics about system logs
        
        Returns:
            Dict: Dictionary containing log statistics
        """
        try:
            with self.Session() as session:
                # Get total count
                total_count = session.query(func.count(SystemLog.log_id)).scalar()
                
                # Get counts by event type
                event_counts_query = (
                    session.query(
                        SystemLog.event_type,
                        func.count(SystemLog.log_id).label('count')
                    )
                    .group_by(SystemLog.event_type)
                    .order_by(desc('count'))
                )
                event_counts = {row[0]: row[1] for row in event_counts_query}
                
                # Get counts by day for the last 7 days
                daily_counts_query = (
                    session.query(
                        func.date_trunc('day', SystemLog.timestamp).label('day'),
                        func.count(SystemLog.log_id).label('count')
                    )
                    .group_by('day')
                    .order_by(desc('day'))
                    .limit(7)
                )
                daily_counts = {str(row[0].date()): row[1] for row in daily_counts_query}
                
                return {
                    "total_count": total_count,
                    "event_counts": event_counts,
                    "daily_counts": daily_counts
                }
                
        except Exception as e:
            logger.error(f"Error retrieving log statistics: {e}")
            return {
                "total_count": 0,
                "event_counts": {},
                "daily_counts": {}
            }
    
    def purge_old_logs(self, days_to_keep: int = 30) -> int:
        """
        Delete logs older than the specified number of days
        
        Args:
            days_to_keep: Number of days of logs to keep (default: 30)
            
        Returns:
            int: Number of logs deleted
        """
        try:
            if self.Session is None:
                logger.error("Cannot purge logs: Session is not initialized")
                return 0
                
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.Session() as session:
                # Find logs older than the cutoff date
                deleted_count = session.query(SystemLog).filter(
                    SystemLog.timestamp < cutoff_date
                ).delete()
                
                session.commit()
                logger.info(f"Purged {deleted_count} logs older than {days_to_keep} days")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error purging old logs: {e}")
            return 0

    def log_allocation_change(self, employee_id: str, project_id: str, 
                             allocation_details: Dict[str, Any], 
                             user: Optional[str] = None, 
                             action: str = "UPDATED") -> bool:
        """
        Log a project allocation change event
        
        Args:
            employee_id: ID of the employee
            project_id: ID of the project
            allocation_details: Details of the allocation change
            user: Username who made the change
            action: Type of action (ADDED, UPDATED, REMOVED)
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        description = f"Project allocation {action}: Employee {employee_id} on Project {project_id}"
        details = {
            "employee_id": employee_id,
            "project_id": project_id,
            "action": action,
            "allocation_details": allocation_details,
            "changed_by": user
        }
        return self.log_event("ALLOCATION_CHANGE", description, user, details)
        
    def log_project_change(self, project_id: str, project_name: str,
                          user: Optional[str] = None, 
                          action: str = "ADDED",
                          project_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a project change event (add, update, delete)
        
        Args:
            project_id: ID of the project
            project_name: Name of the project
            user: Username who made the change
            action: Type of action (ADDED, UPDATED, DELETED)
            project_details: Additional details about the project
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        description = f"Project {action}: {project_name} (ID: {project_id})"
        details = {
            "project_id": project_id,
            "project_name": project_name,
            "action": action,
            "changed_by": user
        }
        
        if project_details:
            details["project_details"] = project_details
            
        return self.log_event("PROJECT_CHANGE", description, user, details)

# Create a singleton instance
_activity_logger = None

def get_logger(engine=None):
    """
    Get the singleton ActivityLogger instance
    
    Args:
        engine: SQLAlchemy engine (optional)
        
    Returns:
        ActivityLogger: The singleton instance
    """
    global _activity_logger
    if _activity_logger is None:
        _activity_logger = ActivityLogger(engine)
    return _activity_logger