import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from core.etl import ETLPipeline
from config.config import etl_config, app_config
from logs.activity_logger import get_logger

def render_file_upload(db_pool):
    """Render the file upload page"""
    st.subheader("File Upload")
    
    # Get the global activity logger instance
    logger = get_logger()

    # Create a single file uploader for multiple files
    uploaded_files = st.file_uploader(
        "Upload CSV Files",
        type=['csv'],
        accept_multiple_files=True,
        help="Drag and drop CSV files here or click to browse"
    )

    if uploaded_files:
        # Process uploaded files
        files_dict = {}
        unrecognized_files = []

        # Process uploaded files silently
        for uploaded_file in uploaded_files:
            # Map file to type based on name (updated mapping)
            file_type = None
            filename_lower = uploaded_file.name.lower()
            
            # More flexible file type detection
            if 'attendance' in filename_lower or 'attendance_report' in filename_lower:
                file_type = 'attendance_report'
            elif 'exit' in filename_lower or 'employee_exit' in filename_lower:
                file_type = 'employee_exit'
            elif 'work' in filename_lower or 'work_profile' in filename_lower or 'employee_work' in filename_lower:
                file_type = 'work_profile'
            elif 'employee_master' in filename_lower or 'master' in filename_lower:
                file_type = 'employee_master'
            elif 'experience' in filename_lower or 'experience_report' in filename_lower:
                file_type = 'experience_report'
            elif 'timesheet' in filename_lower:
                file_type = 'timesheet_report'
            elif 'allocation' in filename_lower or 'project_allocation' in filename_lower:
                file_type = 'project_allocations'
            elif 'resource' in filename_lower or 'utilization' in filename_lower:
                file_type = 'resource_utilization'

            if file_type:
                # Save uploaded file
                save_path = app_config.upload_folder / f"{file_type}_{uploaded_file.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_dict[file_type] = save_path
                # Log file upload
                logger.log_file_upload(
                    filename=uploaded_file.name,
                    file_type=file_type,
                    user=st.session_state.get('username', 'anonymous'),
                    status="SUCCESS",
                    details={
                        "user_full_name": st.session_state.get('user_full_name', 'Unknown'),
                        "upload_time": str(datetime.now())
                    }
                )
            else:
                unrecognized_files.append(uploaded_file.name)
                # Log unrecognized file
                logger.log_event(
                    event_type="FILE_UPLOAD",
                    description=f"Unrecognized file: {uploaded_file.name}",
                    user=st.session_state.get('username', 'anonymous'),
                    details={
                        "status": "FAILED", 
                        "reason": "Unrecognized file type",
                        "user_full_name": st.session_state.get('user_full_name', 'Unknown'),
                        "upload_time": str(datetime.now())
                    }
                )

        # Show simple status message
        if files_dict:
            if unrecognized_files:
                st.warning(f"✓ {len(files_dict)} files uploaded successfully. Unable to process: {', '.join(unrecognized_files)}")
            else:
                st.success(f"✓ All {len(files_dict)} files uploaded successfully!")

            # Process button
            if st.button("Process Files", type="primary"):
                try:
                    with st.spinner("Processing files..."):
                        # Initialize and run ETL pipeline
                        pipeline = ETLPipeline()
                        success, message, stats = pipeline.process_files(files_dict)

                        # Log file processing results
                        for file_type, file_path in files_dict.items():
                            logger.log_file_processing(
                                filename=file_path.name,
                                records_processed=stats.get('records_processed', 0),
                                records_success=stats.get('records_success', 0),
                                records_failed=stats.get('validation_errors', {}).get(file_type, 0),
                                user=st.session_state.get('username', 'anonymous')
                            )

                        # Show simple results
                        if success:
                            st.success("✓ All files processed successfully!")
                        else:
                            st.warning(f"⚠ Processing completed with errors: {message}")

                        # Show error details if any
                        total_errors = sum(stats.get('validation_errors', {}).values())
                        if total_errors > 0:
                            with st.expander("View Error Details"):
                                for stage, count in stats.get('validation_errors', {}).items():
                                    if count > 0:
                                        st.write(f"**{stage.title()} Stage Errors:** {count}")
                                st.write(f"**Message:** {message}")

                except Exception as e:
                    st.error(f"Error processing files: {e}")
                    # Log processing error
                    logger.log_event(
                        event_type="FILE_PROCESSING",
                        description=f"Error processing files: {str(e)}",
                        user=st.session_state.get('user', 'anonymous'),
                        details={"status": "FAILED", "error": str(e)}
                    )

    # Show upload history
    st.subheader("Upload History")
    try:
        with db_pool.get_cursor() as cursor:
            cursor.execute("""
                SELECT upload_id, file_type, upload_timestamp, status,
                       records_processed, records_success, records_failed
                FROM csv_upload_log
                ORDER BY upload_timestamp DESC
                LIMIT 5
            """)
            uploads = cursor.fetchall()

            if uploads:
                df = pd.DataFrame(uploads, columns=[
                    'Upload ID', 'Files', 'Timestamp', 'Status',
                    'Processed', 'Success', 'Failed'
                ])
                st.dataframe(df, use_container_width=True)

                # Show validation errors for selected upload
                selected_upload = st.selectbox(
                    "Select upload to view errors",
                    options=[row[0] for row in uploads],
                    format_func=lambda x: f"Upload {x}"
                )

                cursor.execute("""
                    SELECT field_name, field_value, error_message
                    FROM data_validation_errors
                    WHERE upload_id = %s
                """, (selected_upload,))
                errors = cursor.fetchall()

                if errors:
                    with st.expander("View Validation Errors"):
                        error_df = pd.DataFrame(errors, columns=[
                            'Field', 'Value', 'Error'
                        ])
                        st.dataframe(error_df, use_container_width=True)
            else:
                st.info("No previous uploads found")

    except Exception as e:
        st.error(f"Error loading upload history: {e}")
        # Log error loading upload history
        logger.log_event(
            event_type="ERROR",
            description=f"Error loading upload history: {str(e)}",
            user=st.session_state.get('user', 'anonymous'),
            details={"error": str(e)}
        )