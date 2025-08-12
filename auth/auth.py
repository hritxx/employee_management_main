import streamlit as st
import os
import hashlib
from datetime import datetime
from typing import Optional

class AuthManager:
    """Handles authentication for the Streamlit app"""
    
    def __init__(self):
        self.valid_username = os.getenv('APP_USERNAME')
        self.valid_password = os.getenv('APP_PASSWORD')
        
        # Optional: Use hashed passwords for better security
        self.valid_password_hash = os.getenv('APP_PASSWORD_HASH')
        
    def hash_password(self, password: str) -> str:
        """Create SHA-256 hash of password"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username: str, password: str) -> bool:
        """Verify username and password against environment variables"""
        if not self.valid_username or not (self.valid_password or self.valid_password_hash):
            st.error("Authentication credentials not configured. Please check environment variables.")
            return False
        
        # Check username
        if username != self.valid_username:
            return False
        
        # Check password (plain text or hashed)
        if self.valid_password_hash:
            # Use hashed password verification
            return self.hash_password(password) == self.valid_password_hash
        else:
            # Use plain text password verification
            return password == self.valid_password
    
    def login_form(self) -> bool:
        """Display login form and handle authentication"""
        st.markdown("# Employee Management System")
        st.markdown("### Please log in to access the system")
        
        # Create login form
        with st.form("login_form"):
            st.markdown("#### Login Credentials")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            # Add full name field for tracking changes
            full_name = st.text_input("Full Name", placeholder="Enter your full name (for tracking changes)")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                login_button = st.form_submit_button(" Login", use_container_width=True)
            
            if login_button:
                if not username or not password:
                    st.error("⚠️ Please enter both username and password")
                    return False
                
                if not full_name:
                    st.error("⚠️ Please enter your full name for activity tracking")
                    return False
                
                if self.verify_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    # Save the user's full name in session state for activity logging
                    st.session_state.user_full_name = full_name
                    
                    # Log the login event
                    try:
                        from logs.activity_logger import get_logger
                        logger = get_logger()
                        logger.log_event(
                            event_type="USER_LOGIN",
                            description=f"User logged in: {full_name}",
                            user=username,
                            details={"full_name": full_name, "timestamp": str(datetime.now())}
                        )
                    except Exception as e:
                        # Continue even if logging fails
                        print(f"Failed to log login: {str(e)}")
                    
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                    return True
                else:
                    st.error("❌ Invalid username or password")
                    return False
        
        # Add some styling and information
        st.markdown("---")
        st.markdown("##### Note:")
        st.info("Contact your system administrator if you need access credentials.")
        
        return False
    
    def logout(self):
        """Handle user logout"""
        # Log the logout event
        try:
            if self.is_authenticated():
                from logs.activity_logger import get_logger
                logger = get_logger()
                username = st.session_state.get('username', 'unknown')
                full_name = st.session_state.get('user_full_name', username)
                
                logger.log_event(
                    event_type="USER_LOGOUT",
                    description=f"User logged out: {full_name}",
                    user=username,
                    details={"full_name": full_name, "timestamp": str(datetime.now())}
                )
        except Exception as e:
            # Continue even if logging fails
            print(f"Failed to log logout: {str(e)}")
            
        # Clear session state
        st.session_state.authenticated = False
        if 'username' in st.session_state:
            del st.session_state.username
        if 'user_full_name' in st.session_state:
            del st.session_state.user_full_name
            
        st.rerun()
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return st.session_state.get('authenticated', False)
    
    def get_current_user(self) -> Optional[str]:
        """Get current logged-in username"""
        username = st.session_state.get('username', None)
        full_name = st.session_state.get('user_full_name', None)
        
        if username and full_name:
            return f"{full_name} ({username})"
        return username
    
    def require_auth(self) -> bool:
        """Decorator-like function to require authentication"""
        if not self.is_authenticated():
            return self.login_form()
        return True