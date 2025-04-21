import os
import json
import pandas as pd
import toml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Database paths
DB_DIR = Path("data")
USER_DB_PATH = DB_DIR / "users.json"
CREDIT_DB_PATH = DB_DIR / "credits.json"
CONFIG_PATH = Path("config.toml")

# Ensure database directory exists
os.makedirs(DB_DIR, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "app": {
        "name": "QUACLRS Audio Classifier",
        "default_credits": 100,
        "credits_per_inference": 5,
        "stripe": {
            "test_public_key": "pk_test_51OxTOpHJY6XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "test_secret_key": "sk_test_51OxTOpHJY6XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "credit_price_id": "price_XXXXXXXXXXXXXXXXXXXXXXXX",
            "webhook_secret": "whsec_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        }
    },
    "credit_packages": [
        {"id": "basic", "credits": 100, "price_usd": 5.99, "description": "Basic Package"},
        {"id": "standard", "credits": 500, "price_usd": 19.99, "description": "Standard Package"},
        {"id": "premium", "credits": 2000, "price_usd": 49.99, "description": "Premium Package"}
    ]
}

def init_db():
    """Initialize database files if they don't exist"""
    # Create config file if it doesn't exist
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as f:
            toml.dump(DEFAULT_CONFIG, f)
        logger.info(f"Created config file at {CONFIG_PATH}")
    
    # Create user database if it doesn't exist
    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, 'w') as f:
            json.dump({}, f)
        logger.info(f"Created user database at {USER_DB_PATH}")
    
    # Create credit database if it doesn't exist
    if not os.path.exists(CREDIT_DB_PATH):
        with open(CREDIT_DB_PATH, 'w') as f:
            json.dump({}, f)
        logger.info(f"Created credit database at {CREDIT_DB_PATH}")

def get_config():
    """Get app configuration"""
    if not os.path.exists(CONFIG_PATH):
        init_db()
    
    with open(CONFIG_PATH, 'r') as f:
        return toml.load(f)

def get_users():
    """Get all users from database"""
    if not os.path.exists(USER_DB_PATH):
        init_db()
    
    with open(USER_DB_PATH, 'r') as f:
        return json.load(f)

def save_users(users):
    """Save users to database"""
    with open(USER_DB_PATH, 'w') as f:
        json.dump(users, f, indent=2)

def get_user(email):
    """Get user by email"""
    users = get_users()
    return users.get(email)

def save_user(email, user_data):
    """Save a user to the database"""
    users = get_users()
    users[email] = user_data
    save_users(users)

def get_user_credits():
    """Get all user credits"""
    if not os.path.exists(CREDIT_DB_PATH):
        init_db()
    
    with open(CREDIT_DB_PATH, 'r') as f:
        return json.load(f)

def save_user_credits(credits_data):
    """Save user credits"""
    with open(CREDIT_DB_PATH, 'w') as f:
        json.dump(credits_data, f, indent=2)

def get_user_credit_balance(email):
    """Get credit balance for a user"""
    credits = get_user_credits()
    return credits.get(email, get_config()["app"]["default_credits"])

def update_user_credits(email, new_balance):
    """Update user's credit balance"""
    credits = get_user_credits()
    credits[email] = new_balance
    save_user_credits(credits)

def deduct_credits(email, amount):
    """Deduct credits from a user's balance"""
    current_balance = get_user_credit_balance(email)
    if current_balance >= amount:
        new_balance = current_balance - amount
        update_user_credits(email, new_balance)
        return True, new_balance
    return False, current_balance

def add_credits(email, amount):
    """Add credits to a user's balance"""
    current_balance = get_user_credit_balance(email)
    new_balance = current_balance + amount
    update_user_credits(email, new_balance)
    return new_balance

def get_credit_packages():
    """Get available credit packages"""
    config = get_config()
    return config["credit_packages"]
