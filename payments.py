import stripe
import streamlit as st
from db import get_config, add_credits, get_credit_packages
import logging
import uuid
import time

logger = logging.getLogger(__name__)

# Flag to determine if we should use real Stripe API or simulation
USE_REAL_STRIPE = False

def init_stripe():
    """Initialize Stripe with API keys from config"""
    config = get_config()
    stripe_config = config["app"]["stripe"]
    
    if USE_REAL_STRIPE:
        stripe.api_key = stripe_config["test_secret_key"]
    
    return stripe_config

def create_checkout_session(user_email, package_id):
    """Create a Stripe checkout session for a credit package"""
    stripe_config = init_stripe()
    
    # Get the package details
    packages = get_credit_packages()
    selected_package = None
    
    for package in packages:
        if package["id"] == package_id:
            selected_package = package
            break
    
    if not selected_package:
        logger.error(f"Package not found: {package_id}")
        return None
    
    # If using real Stripe
    if USE_REAL_STRIPE:
        try:
            # Create a checkout session
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[
                    {
                        "price_data": {
                            "currency": "usd",
                            "product_data": {
                                "name": f"{selected_package['description']} - {selected_package['credits']} Credits",
                            },
                            "unit_amount": int(selected_package["price_usd"] * 100),  # Convert to cents
                        },
                        "quantity": 1,
                    },
                ],
                mode="payment",
                success_url=f"http://localhost:8501/success?session_id={{CHECKOUT_SESSION_ID}}&package_id={package_id}&email={user_email}",
                cancel_url="http://localhost:8501/cancel",
                metadata={
                    "user_email": user_email,
                    "package_id": package_id,
                    "credits": selected_package["credits"]
                }
            )
            return checkout_session
        except Exception as e:
            logger.error(f"Error creating checkout session: {str(e)}")
            return None
    
    # Simulated payment flow for demo purposes
    else:
        try:
            # Create a simulated checkout session
            simulated_session = {
                "id": f"sim_session_{uuid.uuid4().hex}",
                "url": f"?success=true&simulated=true&package_id={package_id}&email={user_email}&credits={selected_package['credits']}",
                "metadata": {
                    "user_email": user_email,
                    "package_id": package_id,
                    "credits": selected_package["credits"]
                }
            }
            
            # Store in session state for simulation
            if "simulated_payments" not in st.session_state:
                st.session_state.simulated_payments = {}
            
            st.session_state.simulated_payments[simulated_session["id"]] = {
                "status": "pending",
                "metadata": simulated_session["metadata"],
                "created": time.time()
            }
            
            return type('obj', (object,), simulated_session)
        except Exception as e:
            logger.error(f"Error creating simulated checkout session: {str(e)}")
            return None

def process_successful_payment(session_id):
    """Process a successful payment"""
    # If using simulated payments
    if not USE_REAL_STRIPE and "simulated_payments" in st.session_state:
        # Check if this is a simulated payment
        if session_id in st.session_state.simulated_payments:
            payment_data = st.session_state.simulated_payments[session_id]
            payment_data["status"] = "paid"
            
            user_email = payment_data["metadata"]["user_email"]
            credits = int(payment_data["metadata"]["credits"])
            
            # Add credits to user's account
            new_balance = add_credits(user_email, credits)
            logger.info(f"Added {credits} credits to {user_email}. New balance: {new_balance}")
            return True, f"Successfully added {credits} credits!"
        
        # Check if this is a direct simulation from URL params
        query_params = st.query_params
        if "simulated" in query_params and query_params["simulated"] == "true":
            user_email = query_params.get("email", "")
            credits = int(query_params.get("credits", 0))
            
            if user_email and credits > 0:
                # Add credits to user's account
                new_balance = add_credits(user_email, credits)
                logger.info(f"Added {credits} credits to {user_email}. New balance: {new_balance}")
                return True, f"Successfully added {credits} credits!"
    
    # Real Stripe integration
    elif USE_REAL_STRIPE:
        try:
            stripe_config = init_stripe()
            session = stripe.checkout.Session.retrieve(session_id)
            
            # Verify payment was successful
            if session.payment_status == "paid":
                user_email = session.metadata.get("user_email")
                credits = int(session.metadata.get("credits", 0))
                
                if user_email and credits > 0:
                    # Add credits to user's account
                    new_balance = add_credits(user_email, credits)
                    logger.info(f"Added {credits} credits to {user_email}. New balance: {new_balance}")
                    return True, f"Successfully added {credits} credits!"
                else:
                    logger.error(f"Invalid metadata: {session.metadata}")
                    return False, "Error processing payment: Invalid metadata"
            else:
                logger.error(f"Payment not completed: {session.payment_status}")
                return False, "Payment not completed"
        except Exception as e:
            logger.error(f"Error processing payment: {str(e)}")
            return False, f"Error processing payment: {str(e)}"
    
    return False, "Payment processing failed"

# For simulated flow - process payment immediately
def simulate_payment(session_id, credits, user_email):
    """Simulate a successful payment"""
    new_balance = add_credits(user_email, credits)
    logger.info(f"[SIMULATION] Added {credits} credits to {user_email}. New balance: {new_balance}")
    return new_balance

# For testing only - provide a dummy credit card
def get_test_card_info():
    """Get test card information for Stripe testing"""
    return {
        "number": "4242 4242 4242 4242",
        "exp_month": "12",
        "exp_year": "2025",
        "cvc": "123",
        "name": "Test User",
        "address": "123 Test St, Test City, 12345"
    }
