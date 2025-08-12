"""
Database connection handler with connection pooling
"""

import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Generator
import logging
import os
from config.config import db_config, DATABASE_URL

logger = logging.getLogger(__name__)

class DatabasePool:
    """Database connection pool manager"""
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self):
        """Initialize the connection pool"""
        try:
            if DATABASE_URL:
                self._pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=DATABASE_URL
                )
                logger.info("Database connection pool initialized using DATABASE_URL")
            else:
                self._pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=db_config.host,
                    port=db_config.port,
                    database=db_config.database,
                    user=db_config.user,
                    password=db_config.password
                )
                logger.info("Database connection pool initialized using config values")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator:
        """Get a connection from the pool"""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self) -> Generator:
        """Get a cursor from a pooled connection"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database operation failed: {e}")
                raise
            finally:
                cursor.close()

    def close_all(self):
        """Close all connections in the pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("All database connections closed")

# Global database pool instance
db_pool = DatabasePool()

def get_connection():
    """Module-level function that returns the connection context manager"""
    return db_pool.get_connection()

def get_cursor():
    """Module-level function that returns the cursor context manager"""
    return db_pool.get_cursor()