import streamlit as st
import time
from db import add_credits

def simulated_stripe_checkout(package, user_email):
    """
    Simulates a Stripe checkout experience with form fields
    that look like the real Stripe checkout flow
    """
    st.markdown("""
    <style>
        .stripe-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .stripe-header img {
            height: 40px;
            margin-bottom: 10px;
        }
        .checkout-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            max-width: 550px;
            margin: 0 auto;
        }
        .checkout-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .checkout-price {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .checkout-details {
            color: #666;
            margin-bottom: 25px;
            font-size: 14px;
        }
        .checkout-divider {
            border-top: 1px solid #eee;
            margin: 20px 0;
        }
        .field-label {
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 5px;
            color: #555;
        }
        .card-icon {
            display: inline-block;
            margin-right: 8px;
            vertical-align: middle;
        }
        .secure-badge {
            display: flex;
            align-items: center;
            font-size: 13px;
            color: #666;
            margin-top: 20px;
        }
        .secure-badge svg {
            margin-right: 8px;
        }
        .powered-by {
            text-align: center;
            margin-top: 20px;
            font-size: 13px;
            color: #888;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header with Stripe-like layout
    st.markdown("""
    <div class="stripe-header">
        <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/stripe_logo_icon_167962.png" alt="Stripe">
    </div>
    """, unsafe_allow_html=True)
    
    # Checkout container
    st.markdown(f"""
    <div class="checkout-card">
        <div class="checkout-title">{package['description']}</div>
        <div class="checkout-price">${package['price_usd']}</div>
        <div class="checkout-details">
            You're purchasing {package['credits']} credits for QUACLRS Audio Classifier
        </div>
        <div class="checkout-divider"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Card information form
    with st.form("payment_form"):
        st.subheader("Card Information")
        
        # Card number
        col1, col2 = st.columns([3, 1])
        with col1:
            card_number = st.text_input("Card number", value="4242 4242 4242 4242")
        with col2:
            card_cvc = st.text_input("CVC", value="123")
        
        # Expiration date
        col1, col2 = st.columns(2)
        with col1:
            card_exp_month = st.selectbox("Expiration month", 
                                        options=[f"{i:02d}" for i in range(1, 13)], 
                                        index=4)
        with col2:
            current_year = 2025
            card_exp_year = st.selectbox("Expiration year", 
                                       options=[str(current_year + i) for i in range(10)],
                                       index=2)
        
        # Billing information
        st.subheader("Billing Information")
        
        name = st.text_input("Name on card", value=user_email.split('@')[0].title())
        
        # Country and postal code
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("Country", 
                                 options=["United States", "Canada", "United Kingdom", "Australia", "Germany", "France"],
                                 index=0)
        with col2:
            postal_code = st.text_input("ZIP / Postal code", value="10001")
            
        # Add secure payment message with lock icon
        st.markdown("""
        <div class="secure-badge">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M13 5H12V3.5C12 1.57 10.43 0 8.5 0C6.57 0 5 1.57 5 3.5V5H4C2.9 5 2 5.9 2 7V13C2 14.1 2.9 15 4 15H13C14.1 15 15 14.1 15 13V7C15 5.9 14.1 5 13 5ZM6.5 3.5C6.5 2.4 7.4 1.5 8.5 1.5C9.6 1.5 10.5 2.4 10.5 3.5V5H6.5V3.5ZM13.5 13C13.5 13.28 13.28 13.5 13 13.5H4C3.72 13.5 3.5 13.28 3.5 13V7C3.5 6.72 3.72 6.5 4 6.5H13C13.28 6.5 13.5 6.72 13.5 7V13ZM8.5 11.5C9.33 11.5 10 10.83 10 10C10 9.17 9.33 8.5 8.5 8.5C7.67 8.5 7 9.17 7 10C7 10.83 7.67 11.5 8.5 11.5Z" fill="#555"/>
            </svg>
            Secure payment processed by Stripe
        </div>
        """, unsafe_allow_html=True)
        
        # Submit button
        submitted = st.form_submit_button("Pay $" + str(package['price_usd']), use_container_width=True)
        
        if submitted:
            # Display processing animation
            with st.spinner("Processing your payment..."):
                # Simulate network delay
                time.sleep(2)
                
                # Validate card (simple simulation)
                valid = True
                error_message = None
                
                # Check for valid card number (just for simulation)
                if not card_number.replace(" ", "").isdigit() or len(card_number.replace(" ", "")) != 16:
                    valid = False
                    error_message = "Invalid card number. Please enter a 16-digit card number."
                
                # Process the payment if valid
                if valid:
                    # Add credits to the user's account
                    new_balance = add_credits(user_email, package['credits'])
                    
                    # Show success message
                    st.success(f"Payment successful! {package['credits']} credits have been added to your account. Your new balance is {new_balance} credits.")
                    
                    # Offer to return to the app
                    st.info("You'll be redirected back to the app in a moment...")
                    time.sleep(2)
                    
                    # Return true to indicate successful payment
                    return True
                else:
                    # Show error
                    st.error(error_message)
                    return False
    
    # Powered by Stripe
    st.markdown("""
    <div class="powered-by">
        Powered by <strong>Stripe</strong>
    </div>
    """, unsafe_allow_html=True)
    
    return False
