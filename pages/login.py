import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import login_user, register_user, is_authenticated
import logging

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="QUACLRS - Login",
    page_icon="🎵",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Redirect if already logged in
    if is_authenticated():
        st.switch_page("app.py")
    
    # Logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>QUACLRS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>AI-Powered Audio Classification</p>", unsafe_allow_html=True)
    
    # Auth container
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    
    # Tabs for login and signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form", clear_on_submit=False):
            st.subheader("Welcome Back")
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", placeholder="********")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if not email or not password:
                    st.error("Please enter your email and password")
                else:
                    if login_user(email, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
    
    with tab2:
        with st.form("signup_form", clear_on_submit=True):
            st.subheader("Create an Account")
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", placeholder="********")
            password_confirm = st.text_input("Confirm Password", type="password", placeholder="********")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit_button:
                if not name or not email or not password:
                    st.error("Please fill in all fields")
                elif password != password_confirm:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(email, name, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<p style='text-align: center; margin-top: 2rem; color: #888;'>© 2025 QUACLRS - AI Audio Classification</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
