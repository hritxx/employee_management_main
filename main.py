from core.database import DatabasePool, get_connection, get_cursor
from core.data_seeder import DatabaseSeeder2
from core.etl import ETLPipeline


# This file can be run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

if __name__ == "__main__":
    db_pool = DatabasePool()
    
    # Initialize ETL pipeline and process data
    etl = ETLPipeline()
    processed_data = etl.process_all_files()
    
    # Initialize data seeder and seed data
    with get_connection() as conn:
        with conn.cursor() as cursor:
            seeder = DatabaseSeeder2(conn, cursor)
            seeder.seed_all(processed_data)