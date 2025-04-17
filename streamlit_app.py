import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
from model import HeartDiseaseModel
import json
import base64

# Load custom CSS
def load_css():
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Add custom Google Fonts
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

def add_logo():
    st.markdown("""
        <div class="logo-container">
            <i class="fas fa-heartbeat"></i>
            <span>HeartCare AI</span>
        </div>
    """, unsafe_allow_html=True)

def add_hero_section():
    st.markdown("""
        <div class="hero-container">
            <div class="hero-content">
                <div class="hero-text">
                    <div class="hero-subtitle">Advanced Federated Learning Platform</div>
                    <h1>Collaborative Medical AI for Better Healthcare</h1>
                    <p class="hero-description">
                        Join our federated learning network to develop and improve AI models for various medical conditions.
                        Contribute to breakthrough discoveries while maintaining data privacy and security.
                    </p>
                </div>
                <div class="hero-image">
                    <img src="https://img.freepik.com/free-vector/medical-technology-science-background-vector-health-digital-remix_53876-117739.jpg" 
                         alt="Medical AI Technology Illustration">
                </div>
            </div>
            <svg class="wave-bottom" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 120">
                <path fill="#ffffff" fill-opacity="1" d="M0,32L60,42.7C120,53,240,75,360,74.7C480,75,600,53,720,48C840,43,960,53,1080,58.7C1200,64,1320,64,1380,64L1440,64L1440,120L1380,120C1320,120,1200,120,1080,120C960,120,840,120,720,120C600,120,480,120,360,120C240,120,120,120,60,120L0,120Z"></path>
            </svg>
        </div>
    """, unsafe_allow_html=True)

def add_navigation():
    st.markdown("""
        <nav class="navigation">
            <div class="nav-links">
                <a href="#" class="active">Home</a>
                <a href="#">About Us</a>
                <a href="#">Services</a>
                <a href="#">Contact</a>
            </div>
            <div class="nav-contact">
                <span class="hotline">
                    <i class="fas fa-phone-alt"></i>
                    Hotline: 1-800-HEART
                </span>
            </div>
        </nav>
    """, unsafe_allow_html=True)

def add_features_section():
    st.markdown("""
        <div class="features-container">
            <div class="feature-card">
                <i class="fas fa-network-wired"></i>
                <h3>Federated Learning</h3>
                <p>Train AI models collaboratively while keeping sensitive medical data secure and private on local systems.</p>
            </div>
            <div class="feature-card">
                <i class="fas fa-user-shield"></i>
                <h3>Privacy Focused</h3>
                <p>Advanced privacy-preserving techniques ensure your medical data never leaves your local environment.</p>
            </div>
            <div class="feature-card">
                <i class="fas fa-brain"></i>
                <h3>Medical AI Models</h3>
                <p>Develop and improve various medical AI models through collaborative learning across institutions.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="Medical Federated Learning Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'local_model' not in st.session_state:
    st.session_state.local_model = None
if 'page' not in st.session_state:
    st.session_state.page = "login"

# Database functions
def init_db():
    conn = sqlite3.connect('federated_learning.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create user_actions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action_type TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('federated_learning.db')
    c = conn.cursor()
    
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if not result or not check_password_hash(result[1], password):
        return None
    
    return result[0]

def create_user(username, email, password):
    try:
        conn = sqlite3.connect('federated_learning.db')
        c = conn.cursor()
        
        # Check if username or email already exists
        c.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if c.fetchone():
            return None, "Username or email already exists"
        
        # Create new user
        password_hash = generate_password_hash(password)
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                 (username, email, password_hash))
        
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        
        return user_id, None
        
    except Exception as e:
        return None, str(e)

def get_user_profile(user_id):
    conn = sqlite3.connect('federated_learning.db')
    c = conn.cursor()
    
    c.execute('SELECT id, username, email, created_at FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        return None
    
    return {
        'id': user[0],
        'username': user[1],
        'email': user[2],
        'created_at': user[3]
    }

def update_profile(user_id, email):
    try:
        conn = sqlite3.connect('federated_learning.db')
        c = conn.cursor()
        
        c.execute('UPDATE users SET email = ? WHERE id = ?', (email, user_id))
        conn.commit()
        conn.close()
        return True
        
    except Exception:
        return False

def log_user_action(user_id, action_type, details=None):
    try:
        conn = sqlite3.connect('federated_learning.db')
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO user_actions (user_id, action_type, details)
            VALUES (?, ?, ?)
        ''', (user_id, action_type, details))
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error logging user action: {str(e)}")

# Initialize database
init_db()

def login_page():
    add_logo()
    add_navigation()
    add_hero_section()
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h2 class='form-title'>Login to Your Account</h2>", unsafe_allow_html=True)
            
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    user_id = verify_user(username, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_id = user_id
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            st.markdown("""
                <div class="register-prompt">
                    <p>Don't have an account?</p>
                    <button class="register-link" onclick="handleRegisterClick()">
                        Register Here
                    </button>
                </div>
            """, unsafe_allow_html=True)
            
            # Hidden button for JavaScript interaction
            if st.button("Register", key="register_button", help=None):
                st.session_state.page = "register"
                st.rerun()
    
    add_features_section()

def register_page():
    add_logo()
    add_navigation()
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h2 class='form-title'>Create Your Account</h2>", unsafe_allow_html=True)
            
            with st.form("register_form"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Register")
                
                if submit:
                    if not all([username, email, password, confirm_password]):
                        st.error("All fields are required")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        user_id, error = create_user(username, email, password)
                        if user_id:
                            st.success("Registration successful! Please login.")
                            st.session_state.page = "login"
                            st.rerun()
                        else:
                            st.error(error)
            
            st.markdown("""
                <div class="login-prompt">
                    <p>Already have an account?</p>
                    <button class="login-link" onclick="handleLoginClick()">
                        Login Here
                    </button>
                </div>
            """, unsafe_allow_html=True)
            
            # Hidden button for JavaScript interaction
            if st.button("Login", key="login_button", help=None):
                st.session_state.page = "login"
                st.rerun()

def profile_page():
    st.title("Profile")
    
    user_data = get_user_profile(st.session_state.user_id)
    if not user_data:
        st.error("Could not load profile data")
        return
    
    st.write(f"Username: {user_data['username']}")
    
    with st.form("update_profile"):
        new_email = st.text_input("Email", value=user_data['email'])
        submit = st.form_submit_button("Update Profile")
        
        if submit:
            if update_profile(st.session_state.user_id, new_email):
                st.success("Profile updated successfully")
                log_user_action(st.session_state.user_id, "profile_update", f"Updated email to {new_email}")
            else:
                st.error("Failed to update profile")

def upload_data_page():
    st.title("Upload Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if the DataFrame has all required columns
            required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.write("The CSV file must contain the following columns:")
                st.write(required_columns)
                return
            
            # Display preview of the data
            st.write("Preview of uploaded data:")
            st.write(df.head())
            
            # Initialize model with the data
            try:
                st.session_state.local_model = HeartDiseaseModel(df)
                st.success("Data uploaded and model initialized successfully!")
                log_user_action(st.session_state.user_id, "data_upload", f"Uploaded file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error initializing model: {str(e)}")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your CSV file is properly formatted and contains all required columns.")

def prediction_page():
    st.title("Make Prediction")
    
    if st.session_state.local_model is None:
        st.warning("Please upload data first to initialize the model.")
        return
    
    st.write("Enter patient information:")
    
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=0, max_value=120)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300)
        chol = st.number_input("Cholesterol", min_value=0, max_value=1000)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate", min_value=0, max_value=300)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        
        submit = st.form_submit_button("Make Prediction")
        
        if submit:
            # Convert categorical inputs to numerical
            sex_map = {"Male": 1, "Female": 0}
            cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            fbs_map = {"Yes": 1, "No": 0}
            restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            exang_map = {"Yes": 1, "No": 0}
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
            
            # Create input data
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex_map[sex]],
                'cp': [cp_map[cp]],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [fbs_map[fbs]],
                'restecg': [restecg_map[restecg]],
                'thalach': [thalach],
                'exang': [exang_map[exang]],
                'oldpeak': [oldpeak],
                'slope': [slope_map[slope]],
                'ca': [ca],
                'thal': [thal_map[thal]]
            })
            
            try:
                prediction = st.session_state.local_model.predict(input_data)
                probability = st.session_state.local_model.predict_proba(input_data)
                
                if prediction[0] == 1:
                    st.error(f"Heart Disease Detected (Probability: {probability[0][1]:.2%})")
                else:
                    st.success(f"No Heart Disease Detected (Probability: {probability[0][0]:.2%})")
                
                log_user_action(
                    st.session_state.user_id,
                    "prediction",
                    f"Prediction made: {prediction[0]} (Probability: {max(probability[0]):.2%})"
                )
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
            <div style='text-align: center; padding: 1rem;'>
                <h3>Welcome, {st.session_state.username}! 👋</h3>
            </div>
        """, unsafe_allow_html=True)
        
        navigation = st.radio(
            "",
            ["Profile", "Upload Data", "Make Prediction", "Logout"],
            key="nav"
        )
        
        return navigation

def main():
    if not st.session_state.logged_in:
        if st.session_state.page == "register":
            register_page()
        else:
            login_page()
    else:
        # Always show sidebar when logged in
        navigation = show_sidebar()
        
        # Main content area
        if navigation == "Profile":
            profile_page()
        elif navigation == "Upload Data":
            upload_data_page()
        elif navigation == "Make Prediction":
            prediction_page()
        elif navigation == "Logout":
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.session_state.local_model = None
            st.session_state.page = "login"
            st.rerun()

if __name__ == "__main__":
    main() 