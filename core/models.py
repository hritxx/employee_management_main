"""
Database models and table creation
"""

from core.database import db_pool
import logging
from core.tables import DatabaseTableCreator
from config.config import db_config

logger = logging.getLogger(__name__)

def create_tables():
    """Create database tables if they don't exist"""
    try:
        # Use DatabaseTableCreator from tables.py
        creator = DatabaseTableCreator()
        creator.connect_postgresql(
            host=db_config.host,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password,
            port=db_config.port
        )
        creator.create_tables()
        creator.close_connection()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise 