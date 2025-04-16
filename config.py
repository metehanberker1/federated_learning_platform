import os

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')  # Local development server
API_ENDPOINTS = {
    'submit_weights': f'{API_BASE_URL}/submit_weights',
    'get_global_model': f'{API_BASE_URL}/get_global_model',
    'predict': f'{API_BASE_URL}/predict',
    'verify_user': f'{API_BASE_URL}/verify_user',
    'create_user': f'{API_BASE_URL}/create_user',
    'get_user_profile': f'{API_BASE_URL}/get_user_profile',
    'update_profile': f'{API_BASE_URL}/update_profile',
    'get_user_actions': f'{API_BASE_URL}/get_user_actions'
}

# Database Configuration
DB_URL = os.getenv('DATABASE_URL', 'sqlite:///users.db')

# Model Configuration
MODEL_SETTINGS = {
    'min_accuracy': 0.7,
    'max_training_time': 300,  # seconds
    'allowed_file_types': ['csv', 'txt', 'xlsx', 'xls']
}

# Security Configuration
SECURITY_SETTINGS = {
    'allowed_origins': ['http://localhost:8501'],  # Local Streamlit development server
    'required_permissions': ['local-training']  # Simplified permissions for testing
} 