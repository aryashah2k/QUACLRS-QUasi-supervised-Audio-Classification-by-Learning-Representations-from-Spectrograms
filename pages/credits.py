import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import is_authenticated
from db import get_user_credit_balance, get_credit_packages, add_credits
from payments import create_checkout_session, get_test_card_info
from stripe_checkout import simulated_stripe_checkout
import logging

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="QUACLRS - Credits",
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
    .credit-balance {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        margin: 1rem 0;
    }
    .credit-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #eee;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .credit-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .credit-price {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4A90E2;
    }
    .credit-amount {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .credit-button {
        background-color: #4A90E2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .test-card-info {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #4A90E2;
    }
    .checkout-container {
        background-color: #f5f9ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid #4A90E2;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Check if user is authenticated
    if not is_authenticated():
        st.switch_page("pages/login.py")
    
    # Title
    st.markdown("<h1 class='main-header'>Credits Dashboard</h1>", unsafe_allow_html=True)
    
    # Get user's credit balance
    credit_balance = get_user_credit_balance(st.session_state.email)
    
    # Display current balance
    st.markdown(f"<div class='credit-balance'>{credit_balance} Credits</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Credits are used for each audio classification</p>", unsafe_allow_html=True)
    
    # Credit packages
    st.subheader("Purchase More Credits")
    
    packages = get_credit_packages()
    
    # Check for success parameter in URL (after Stripe redirect)
    query_params = st.query_params
    if "success" in query_params and query_params["success"] == "true":
        st.success("Payment successful! Credits have been added to your account.")
        
        # If this is a simulated payment, process it immediately
        if "simulated" in query_params and query_params["simulated"] == "true":
            email = query_params.get("email", "")
            package_id = query_params.get("package_id", "")
            credits = int(query_params.get("credits", 0))
            
            if email and credits > 0:
                # Make sure current user is the one getting credits
                if email == st.session_state.email:
                    new_balance = add_credits(email, credits)
                    st.success(f"Added {credits} credits! Your new balance is {new_balance} credits.")
        
        # Clear the query parameters
        st.query_params.clear()
    
    # Check for checkout mode
    if "checkout_mode" not in st.session_state:
        st.session_state.checkout_mode = False
        st.session_state.selected_package = None
    
    # Display package grid if not in checkout mode
    if not st.session_state.checkout_mode:
        # Display credit packages in a grid
        cols = st.columns(3)
        
        for i, package in enumerate(packages):
            with cols[i % 3]:
                st.markdown(f"""
                <div class='credit-card'>
                    <div class='credit-amount'>{package["credits"]} Credits</div>
                    <div class='credit-price'>${package["price_usd"]}</div>
                    <p>{package["description"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Select Package", key=f"buy_{package['id']}"):
                    st.session_state.selected_package = package
                    st.session_state.checkout_mode = True
                    st.rerun()
    
    # Show Stripe checkout if in checkout mode
    else:
        package = st.session_state.selected_package
        
        # Create a back button
        if st.button("← Back to Packages", key="back_button"):
            st.session_state.checkout_mode = False
            st.session_state.selected_package = None
            st.rerun()
        
        # Display the stripe checkout simulation
        payment_success = simulated_stripe_checkout(package, st.session_state.email)
        
        # If payment was successful, return to packages view
        if payment_success:
            st.session_state.checkout_mode = False
            st.session_state.selected_package = None
            st.rerun()
    
    # Transaction History (placeholder for future enhancement)
    with st.expander("Transaction History"):
        st.info("Transaction history will be available in a future update.")

if __name__ == "__main__":
    main()
