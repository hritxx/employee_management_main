
import streamlit as st
import pandas as pd
import json
import requests
import re
from datetime import datetime, date
import logging

# Set up logging
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle timestamps, dates, and NaN values"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def get_database_context(engine, get_available_tables_func):
    """
    Get database schema and sample data for AI context
    
    Args:
        engine: SQLAlchemy engine object
        get_available_tables_func: Function that returns list of available tables
    
    Returns:
        tuple: (table_schema, sample_data)
    """
    table_schema = {}
    sample_data = {}
    available_tables = get_available_tables_func()

    if not available_tables:
        st.warning("No tables found in the database. Please ensure your database is properly set up.")
        return {}, {}

    for table in available_tables:
        try:
            # Get table schema
            schema_query = f"""
                SELECT column_name, data_type, 
                       is_nullable, column_default,
                       character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = '{table}' 
                ORDER BY ordinal_position
            """
            schema_df = pd.read_sql(schema_query, engine)
            table_schema[table] = schema_df.to_dict(orient='records')

            # Get sample data (first 5 rows)
            sample_query = f"SELECT * FROM {table} LIMIT 5"
            sample_df = pd.read_sql(sample_query, engine)
            sample_data[table] = sample_df.to_dict(orient='records')
            
        except Exception as e:
            logger.warning(f"Could not get schema/data for table {table}: {e}")
            continue

    return table_schema, sample_data

def generate_sql_query(user_query, table_schema, sample_data, gemini_api_key):
    """
    Generate SQL query using Gemini API
    
    Args:
        user_query: Natural language query from user
        table_schema: Dictionary containing table schemas
        sample_data: Dictionary containing sample data from tables
        gemini_api_key: Gemini API key
    
    Returns:
        str: Generated SQL query or None if failed
    """
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"

    prompt = f"""
You are a database expert who helps convert natural language queries to SQL queries for PostgreSQL.

Here are the tables in the database:
{json.dumps(table_schema, indent=2, cls=CustomJSONEncoder)}

Here's a sample of each table's data:
{json.dumps(sample_data, indent=2, cls=CustomJSONEncoder)}

The user wants the following information:
{user_query}

IMPORTANT: 
- This is a PostgreSQL database, so use PostgreSQL syntax
- Use ROUND(column::numeric, 2) for rounding numeric values
- Use %s for parameterized queries instead of ?
- All column names are lowercase (e.g., employee_name, not "Employee Name")
- Check which columns exist in which tables before writing queries
- The employee_master table contains employee_name and basic employee information  
- The timesheets table contains project assignments and hours but NOT employee names
- Join tables appropriately to get the information needed
- AVOID using advanced functions like crosstab or PIVOT that require extensions
- For pivot-like operations, use standard aggregations with CASE statements instead
- Keep queries as simple as possible while meeting the requirements

Please generate a valid PostgreSQL SQL query.
Make sure to use DISTINCT or appropriate GROUP BY clauses to avoid duplicate rows.
Return only the SQL query without any explanation or additional text.
Your SQL query should be wrapped in triple backticks like this:
```sql
SELECT * FROM table;
```
"""

    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data['candidates'][0]['content']['parts'][0]['text']
            
            # Extract SQL query from response
            sql_match = re.search(r"```(?:sql)?\n([\s\S]*?)\n```", response_text)
            
            if sql_match:
                return sql_match.group(1).strip()
            else:
                st.error("Could not extract SQL query from API response")
                st.text("Full response:")
                st.text(response_text)
                return None
        else:
            st.error(f"API request failed with status code {response.status_code}")
            st.error(response.text)
            return None
            
    except Exception as e:
        st.error(f"Error making API request: {str(e)}")
        return None

def execute_sql_query(sql_query, engine):
    """
    Execute SQL query and return results
    
    Args:
        sql_query: SQL query string
        engine: SQLAlchemy engine object
    
    Returns:
        pandas.DataFrame: Query results or None if failed
    """
    try:
        result_df = pd.read_sql(sql_query, engine)
        return result_df
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

@st.cache_data
def cached_get_database_context(_engine, _get_available_tables_func):
    """Cached version of get_database_context to improve performance"""
    return get_database_context(_engine, _get_available_tables_func)

def render_ai_query_assistant(engine, get_available_tables_func, gemini_api_key):
    """
    Render the AI Query Assistant interface
    
    Args:
        engine: SQLAlchemy engine object
        get_available_tables_func: Function that returns list of available tables
        gemini_api_key: Gemini API key
    """
    st.header("AI Query Assistant")
    st.info("Describe what you want to find out in plain English, and let AI generate the SQL query for you.")

    user_query = st.text_area(
        "What would you like to know?",
        placeholder="Example: Show me all employees in the Marketing department who worked on Project PRJ003 in June 2025"
    )

    if st.button("Generate Report", key="ai_query_report"):
        if not user_query:
            st.warning("Please enter a query description.")
        else:
            with st.spinner("Generating SQL query..."):
                try:
                    # Get database schema information for context
                    table_schema, sample_data = cached_get_database_context(engine, get_available_tables_func)

                    if not table_schema:
                        st.error("No table information available. Please check your database setup.")
                        return

                    # Generate SQL query using AI
                    sql_query = generate_sql_query(user_query, table_schema, sample_data, gemini_api_key)
                    
                    if sql_query:
                        # Display the generated SQL
                        st.subheader("Generated SQL Query:")
                        st.code(sql_query, language="sql")

                        # Execute the query
                        with st.spinner("Executing query..."):
                            result_df = execute_sql_query(sql_query, engine)
                            
                            if result_df is not None:
                                # Store results and display
                                st.session_state.current_df = result_df
                                st.subheader("Query Results:")
                                st.dataframe(result_df)
                                st.success(f"Query executed successfully! Found {len(result_df)} records.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")