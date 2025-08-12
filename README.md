# Employee Management System

A comprehensive, AI-powered HR management platform built with Python, Streamlit, PostgreSQL/SQLite, and SQLAlchemy. Streamline your workforce management with intelligent automation and intuitive interfaces.

## 🚀 Quick Start

### Step 1: Get the Code
```bash
git clone https://github.com/yamini-Krishna/employee-management.git
cd employee-management
```

### Step 2: Run the Application
```bash
docker-compose up --build
```

### Step 3: Access Your Application
- **Web Interface:** Open http://localhost:8501 in your browser
- **Database:** Available on port 5432 (for admin access if needed)

## 🛠️ Commands You'll Use

### Start the Application
```bash
docker-compose up --build
```
- Builds the app image - employee_management-app 
- Starts all services (app + database)
- Use `--build` to ensure latest changes are included

### Stop the Application
```bash
docker-compose down
```
- Stops all running containers
- Keeps your data safe

### View Running Containers
```bash
docker-compose ps
```

### View Application Logs
```bash
docker-compose logs -f
```

## 🔧 Development Mode

### Make Changes and Test
1. Edit your code files
2. Restart the application:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

### Access Database Directly (if needed)
```bash
docker-compose exec database psql -U your_username -d employee_db
```

##  Troubleshooting

### Port Already in Use
If you get a port error, stop other applications using port 8501:
```bash
docker-compose down
# Wait a few seconds, then try again
docker-compose up --build
```

### Clean Start (Reset Everything)
```bash
docker-compose down -v
docker-compose up --build
```

### Check What's Running
```bash
docker ps
```
---

## ✨ Features

### Core HR Management
- **Employee Master Data** - Complete employee profiles with personal and professional details
- **Project Management** - Track projects, assignments, and deliverables
- **Time Tracking** - Comprehensive timesheet and attendance management
- **Resource Allocation** - Optimize employee-project assignments

### Advanced Capabilities
- **AI-Powered Analytics** - Intelligent task summaries and insights
- **Custom Query Assistant** - Natural language to SQL conversion
- **File Upload & ETL** - Bulk data import with validation
- **Real-time Reporting** - Dynamic dashboards and analytics
- **Audit Trail** - Complete change logging and error tracking

### Technical Features
- **Modular Architecture** - Clean, extensible codebase
- **Multi-Database Support** - PostgreSQL for production, SQLite for development
- **Responsive UI** - Modern Streamlit interface
- **Docker Ready** - Containerized deployment

## 🛠 Prerequisites

- **Python 3.8+** (for local setup)
- **Docker & Docker Compose** (for containerized setup)
- **PostgreSQL** (optional - SQLite included for testing)
- **Internet Connection** (for AI features)

---     
## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
DATABASE_URL=sqlite:///employee_management.db
# For PostgreSQL: postgresql://username:password@localhost:5432/employee_db

# AI Features (Optional)
GEMINI_API_KEY=your_gemini_api_key_here

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
```

### Database Setup

**SQLite (Default):**
- No additional setup required
- Database file created automatically

**PostgreSQL:**
```bash
# Create database
createdb employee_management

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://username:password@localhost:5432/employee_management
```

## 🤖 AI Features Setup

### Gemini API Configuration

1. **Get API Key:**
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in with Google account
   - Click "Create API key"
   - Copy the generated key

2. **Configure the application:**
   ```bash
   # Add to .env/docker-compose.yml file
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Test AI features:**
   - Navigate to "🤖 AI Query Assistant"
   - Try: "Show me all employees hired this year"
   - The AI will generate and execute SQL queries

### AI Features Available

- **Smart Query Assistant** - Natural language to SQL
- **Task Summaries** - AI-generated project insights


## 📁 Project Structure

```
employee_management/
├── __init__.py
├── .env
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose
├── docker-compose.yml
├── main.py
├── requirements.txt
├── sample_queries.md
├── README.md
│
├── auth/
│   └── auth.py
│
├── config/
│   └── config.py
│
├── core/
│   ├── data_seeder.py
│   ├── database.py
│   ├── etl.py
│   ├── models.py
│   └── tables.py
│
├── pages/
│   ├── allocations.py
│   ├── app.py
│   ├── custom_queries.py
│   ├── employee_master.py
│   ├── file_upload.py
│   ├── query_assistant.py
│   ├── report.py
│   ├── summary_reports.py
│   └── tasks_summariser.py
│
├── logs/
│   ├── activity_log_view.py
│   ├── activity_logger.py
│   ├── app.log
│   └── (many db_creation_*.log files)
│
├── csv_files/
│   ├── attendance_report_daily.csv
│   ├── employee_exit_report.csv
│   ├── employee_master.csv
│   ├── employee_work_profile.csv
│   ├── experience_report.csv
│   ├── project_allocations.csv
│   ├── resource_utilization.csv
│   └── timesheet_report_clean.csv
│
├── data/
│   └── (data files, if any)
│
├── uploads/
│   └── (uploaded files, if any)
│
├── init_db/
│   └── (database initialization scripts)
│
└── __pycache__/
    └── (Python cache files)
```

## 🔧 Troubleshooting

### Common Issues

**Application won't start:**
```bash
# Check if port 8501 is available
netstat -an | grep 8501

# Kill existing Streamlit processes
pkill -f streamlit

# Restart with different port
streamlit run pages/app.py --server.port 8502
```

**Database connection errors:**
```bash
# For SQLite - check file permissions
ls -la *.db

# For PostgreSQL - test connection
psql -h localhost -U username -d employee_management
```

**AI features not working:**
- Verify GEMINI_API_KEY is set correctly
- Check internet connection
- Ensure Generative Language API is enabled
- Try regenerating API key

**Docker issues:**
```bash
# Clean rebuild
docker-compose down
docker system prune -f
docker-compose up --build
```




## Acknowledgements

- **[Streamlit](https://streamlit.io/)** - Web framework
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - Database ORM
- **[PostgreSQL](https://www.postgresql.org/)** - Database engine
- **[Google Gemini](https://ai.google.dev/)** - AI capabilities
- **[Docker](https://www.docker.com/)** - Containerization



