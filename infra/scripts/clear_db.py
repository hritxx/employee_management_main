import os
import psycopg2

# Get DB credentials from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5434")  # match docker-compose port mapping
DB_NAME = os.getenv("DB_NAME", "employee_db")  # match docker-compose DB name
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123")  # match docker-compose password

TABLES = [
    "task_summary_history",
    "task_summary",
    "data_validation_errors",
    "csv_upload_log",
    "employee_exit",
    "attendance",
    "timesheet",
    "project_allocation",
    "employee_personal",
    "employee_financial",
    "employee",
    "project",
    "department",
    "designation"
]

def clear_tables():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET session_replication_role = 'replica';")
    for table in TABLES:
        cur.execute(f"DELETE FROM {table};")
        print(f"Cleared table: {table}")
    cur.execute("SET session_replication_role = 'origin';")
    cur.close()
    conn.close()

if __name__ == "__main__":
    clear_tables()