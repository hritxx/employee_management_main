# Employee Management System: Backup & Logging Guide

## Overview

This document provides instructions for using the backup and logging system implemented in the Employee Management System.

## Backup System

### Automated Weekly Backups

- Scheduled to run every Sunday at 2:00 AM
- Backups are stored in `/app/data/backups/` inside the container (mapped to `./data/backups/` on the host)
- Backups are automatically compressed (.gz format)
- Backups older than 30 days are automatically removed

### Manual Backups

1. Navigate to the "Settings" tab in the application
2. Select the "Backup" sub-tab
3. Enter your name (for tracking purposes)
4. Select a reason for the backup
5. Click "Create Backup Now"

### Backup Verification

The system automatically verifies each backup by testing the compressed file integrity. Failed backups are logged and reported.

## Logging System

### User Attribution

All actions in the system are logged with user attribution:
- Username (from login)
- Full name (provided at login)
- Timestamp
- Action details

### Log Types

The system logs the following types of events:
- User logins/logouts
- File uploads and processing
- Project/allocation changes
- Database queries (standard and AI-assisted)
- Backup operations (automatic and manual)
- System maintenance

### Viewing Logs

1. Navigate to the "Settings" tab in the application
2. Select the "Logs" sub-tab
3. Use the filters to view specific log types or search for keywords
4. Multiple views are available (All, User Activity, File Operations, etc.)

### Log Retention

- Logs are automatically purged after 30 days
- Manual purging is available in the Logs > Maintenance tab (admin users only)

## Container Management

### Scripts for Backup and Application Management

The following scripts are available for managing the application and its backup functionality:

1. **docker-backup.sh** - Weekly backup script (used internally by the container)
2. **startup.sh** - Main startup script with backup and logging system
3. **restart.sh** - Restart containers without rebuilding
4. **rebuild.sh** - Rebuild and restart containers (for code changes)

Make sure all scripts have executable permissions:
```bash
chmod +x *.sh
```

### Starting the Application

For normal startup:
```bash
./startup.sh
```
```

### Rebuilding (with code changes)

```bash
./rebuild.sh
```

### Accessing the Application

- Web interface: http://localhost:8503
- Default credentials: See `.env` file or ask administrator

## Troubleshooting

### Docker Build Permission Issues

If you encounter permission errors when building Docker images, such as:
```
failed to solve: error from sender: open /home/yamini/employee_management/data: permission denied
```

This issue is caused by Docker trying to access the data directory during build. The recommended solution is to:

1. Use the updated .dockerignore file (already included) which excludes the data directory during build
2. Let Docker create the data directory with the correct permissions at runtime

If you still encounter permission issues, you can fix them with:
```bash
sudo chown -R $(id -u):$(id -g) ./data
sudo chmod -R 755 ./data
```

### Backup Issues

1. Check if the database container is running and healthy:
   ```bash
   docker ps
   ```
2. Verify permissions on the data directory:
   ```bash
   ls -la ./data
   ```
3. Check backup logs:
   ```bash
   docker logs employee_backup_scheduler
   ```
4. Verify pg_dump is available in the container:
   ```bash
   docker exec -it employee_manager_app pg_dump --version
   ```

### Logging Issues

1. Check if logs are being written:
   ```bash
   docker exec -it employee_manager_app ls -la /app/logs
   ```
2. Verify database connectivity within the app container:
   ```bash
   docker exec -it employee_manager_app python -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:postgres123@postgres-db:5432/employee_db'); print('Connected successfully')"
   ```

## Maintenance

### Database Backups

- Backups are stored in `./data/backups/` on the host
- Consider periodically copying these to external storage for additional protection

### Log Management

- Use the Maintenance tab to purge old logs as needed
- Monitor disk usage regularly

## Contact

For assistance with the backup and logging system, contact:
- System administrator: [YOUR-EMAIL@EXAMPLE.COM]
