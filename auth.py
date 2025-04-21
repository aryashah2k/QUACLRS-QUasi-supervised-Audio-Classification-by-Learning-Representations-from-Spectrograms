import streamlit as st
import jwt
import datetime
import hashlib
import uuid
import re
from db import get_user, save_user
import logging

logger = logging.getLogger(__name__)

# JWT secret key (should be in environment variables for production)
JWT_SECRET = "your-jwt-secret-key-keep-this-secure-in-production"
# Token expiry (24 hours)
TOKEN_EXPIRY = 24 * 60 * 60

def hash_password(password):
    """Hash a password for storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

def validate_password(password):
    """Validate password strength"""
    # At least 8 characters, one uppercase, one lowercase, one number
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$"
    return bool(re.match(pattern, password))

def generate_token(email):
    """Generate JWT token for authentication"""
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=TOKEN_EXPIRY),
        "iat": datetime.datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["email"]
    except:
        return None

def is_authenticated():
    """Check if user is authenticated"""
    if "token" in st.session_state:
        email = verify_token(st.session_state.token)
        if email:
            st.session_state.email = email
            return True
    return False

def login_user(email, password):
    """Login a user"""
    user = get_user(email)
    if user and user["password_hash"] == hash_password(password):
        token = generate_token(email)
        st.session_state.token = token
        st.session_state.email = email
        st.session_state.name = user.get("name", email.split("@")[0])
        return True
    return False

def register_user(email, name, password):
    """Register a new user"""
    # Check if user already exists
    if get_user(email):
        return False, "Email already registered"
    
    # Validate email
    if not validate_email(email):
        return False, "Invalid email format"
    
    # Validate password
    if not validate_password(password):
        return False, "Password must be at least 8 characters and include uppercase, lowercase, and numbers"
    
    # Create user
    user_data = {
        "email": email,
        "name": name,
        "password_hash": hash_password(password),
        "created_at": datetime.datetime.now().isoformat(),
        "user_id": str(uuid.uuid4())
    }
    
    save_user(email, user_data)
    
    # Auto-login after registration
    token = generate_token(email)
    st.session_state.token = token
    st.session_state.email = email
    st.session_state.name = name
    
    return True, "Registration successful"

def logout_user():
    """Logout current user"""
    if "token" in st.session_state:
        del st.session_state.token
    if "email" in st.session_state:
        del st.session_state.email
    if "name" in st.session_state:
        del st.session_state.name
