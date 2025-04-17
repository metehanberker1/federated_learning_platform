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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
def numpy_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json_serializable(item) for item in obj]
    return obj

def init_db():
    conn = sqlite3.connect('federated_learning.db')
    c = conn.cursor()
    
    # Drop existing tables to ensure clean initialization
    c.execute('DROP TABLE IF EXISTS model_contributions')
    c.execute('DROP TABLE IF EXISTS master_model')
    
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
    
    # Create model_contributions table with accuracy
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_contributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            num_samples INTEGER,
            model_weights BLOB,
            accuracy REAL,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create master_model table
    c.execute('''
        CREATE TABLE IF NOT EXISTS master_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_weights BLOB,
            updated_at TIMESTAMP
        )
    ''')
    
    # Initialize master model with test data
    test_data = pd.DataFrame({
        'age': [63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 0],
        'cp': [3, 0, 0, 3, 2],
        'trestbps': [145, 160, 120, 130, 130],
        'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 0, 0, 1, 0],
        'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
        'slope': [0, 1, 1, 0, 2],
        'ca': [0, 3, 2, 0, 0],
        'thal': [1, 2, 2, 2, 2],
        'target': [0, 1, 1, 0, 1]
    })
    
    # Initialize model with test data
    initial_model = HeartDiseaseModel(test_data)
    initial_weights = initial_model.get_weights()
    
    # Convert numpy arrays to JSON serializable format
    serializable_weights = numpy_to_json_serializable(initial_weights)
    
    # Store initial model weights
    c.execute('DELETE FROM master_model')  # Clear any existing entries
    c.execute('''
        INSERT INTO master_model (model_weights, updated_at)
        VALUES (?, ?)
    ''', (json.dumps(serializable_weights), datetime.now()))
    
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
    
    # Display user info
    st.markdown(f"""
        <div class="profile-info">
            <h3>User Information</h3>
            <p><strong>Username:</strong> {user_data['username']}</p>
            <p><strong>Email:</strong> {user_data['email']}</p>
            <p><strong>Member since:</strong> {user_data['created_at']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Get user actions history
    conn = sqlite3.connect('federated_learning.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT action_type, details, timestamp 
        FROM user_actions 
        WHERE user_id = ? 
        AND (action_type = 'data_upload' OR action_type = 'prediction')
        ORDER BY timestamp DESC
    ''', (st.session_state.user_id,))
    
    actions = c.fetchall()
    conn.close()
    
    if actions:
        st.markdown("<h3>Activity History</h3>", unsafe_allow_html=True)
        
        for action in actions:
            action_type, details, timestamp = action
            icon = "📤" if action_type == "data_upload" else "🔍"
            action_name = "Data Upload" if action_type == "data_upload" else "Model Inference"
            
            st.markdown(f"""
                <div class="action-card">
                    <div class="action-header">
                        <span>{icon} {action_name}</span>
                        <span class="action-time">{timestamp}</span>
                    </div>
                    <p class="action-details">{details}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No activity recorded yet")

def upload_data_page():
    st.title("Upload Training Data")
    
    st.write("Upload your training dataset (CSV format with target variables)")
    
    uploaded_training_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        key="training_data_uploader",
        help="Upload a CSV file containing training data with target variables"
    )
    
    if uploaded_training_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_training_file)
            
            # Check if the DataFrame has all required columns
            required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.write("The training CSV file must contain the following columns:")
                st.write(required_columns)
                return
            
            # Display preview of the data
            st.write("Preview of training data:")
            st.write(df.head())
            
            # Initialize and train local model
            try:
                progress_text = "Training local model..."
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Training steps
                status_text.text("Initializing model...")
                progress_bar.progress(10)
                
                local_model = HeartDiseaseModel(df)
                progress_bar.progress(30)
                status_text.text("Training in progress...")
                
                # Get training metrics
                X = df.drop('target', axis=1)
                y = df['target']
                train_predictions = local_model.predict(X)
                
                progress_bar.progress(50)
                status_text.text("Getting model weights...")
                
                # Get local model weights and prepare for storage
                num_samples = len(df)
                model_weights = local_model.get_weights()
                serializable_weights = numpy_to_json_serializable(model_weights)
                
                progress_bar.progress(60)
                status_text.text("Updating federated model...")
                
                # Store the contribution and update master model
                conn = sqlite3.connect('federated_learning.db')
                c = conn.cursor()
                
                try:
                    # Begin transaction
                    c.execute('BEGIN TRANSACTION')
                    
                    # Store the contribution
                    c.execute('''
                        INSERT INTO model_contributions (user_id, num_samples, model_weights, accuracy, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        st.session_state.user_id,
                        num_samples,
                        json.dumps(serializable_weights),
                        0.0,  # Placeholder accuracy value
                        datetime.now()
                    ))
                    
                    progress_bar.progress(70)
                    status_text.text("Performing federated averaging...")
                    
                    # Get all contributions for federated averaging
                    c.execute('''
                        SELECT model_weights, num_samples 
                        FROM model_contributions 
                        ORDER BY timestamp DESC
                    ''')
                    contributions = c.fetchall()
                    
                    # Calculate total samples across all contributions
                    total_samples = sum(contrib[1] for contrib in contributions)
                    
                    # Initialize dictionary for averaged weights
                    averaged_weights = {}
                    
                    # Process first contribution to set up the structure
                    first_weights = json.loads(contributions[0][0])
                    for key in first_weights:
                        averaged_weights[key] = np.zeros_like(np.array(first_weights[key]))
                    
                    # Perform federated averaging
                    for weights_json, num_samples in contributions:
                        weights = json.loads(weights_json)
                        weight = num_samples / total_samples
                        for key in weights:
                            averaged_weights[key] += np.array(weights[key]) * weight
                    
                    progress_bar.progress(80)
                    status_text.text("Saving master model...")
                    
                    # Convert averaged weights to JSON serializable format
                    serializable_averaged_weights = numpy_to_json_serializable(averaged_weights)
                    
                    # Update master model
                    c.execute('''
                        INSERT INTO master_model (model_weights, updated_at)
                        VALUES (?, ?)
                    ''', (json.dumps(serializable_averaged_weights), datetime.now()))
                    
                    # Commit transaction
                    c.execute('COMMIT')
                    
                except Exception as e:
                    # Rollback in case of error
                    c.execute('ROLLBACK')
                    raise e
                
                finally:
                    conn.close()
                
                progress_bar.progress(100)
                status_text.text("Training complete!")
                
                # Log the training contribution
                log_user_action(
                    st.session_state.user_id,
                    "data_upload",
                    f"Contributed training data with {num_samples} samples"
                )
                
                st.success(f"""
                    Local model training complete! Your contribution has been added to the federated learning system.
                    - Number of samples: {num_samples}
                    - Model weights have been aggregated with the master model
                """)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your CSV file is properly formatted and contains all required columns.")

def prediction_page():
    st.title("Make Predictions")
    
    st.write("Upload a dataset for prediction (CSV format)")
    
    test_file = st.file_uploader(
        "Choose a CSV file for prediction", 
        type="csv",
        key="test_data_uploader",
        help="Upload a CSV file containing test data"
    )
    
    if test_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(test_file)
            
            # Check if the DataFrame has all required columns except 'target'
            required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.write("The prediction CSV file must contain the following columns:")
                st.write(required_columns)
                return
            
            # Remove target column if present
            if 'target' in df.columns:
                df = df.drop('target', axis=1)
            
            # Get latest master model weights
            conn = sqlite3.connect('federated_learning.db')
            c = conn.cursor()
            
            c.execute('SELECT model_weights FROM master_model ORDER BY updated_at DESC LIMIT 1')
            result = c.fetchone()
            conn.close()
            
            if result is None:
                st.error("No master model available. Please contact the administrator.")
                return
            
            # Initialize model with test data (required for the model to work)
            test_data = pd.DataFrame({
                'age': [63, 67, 67, 37, 41],
                'sex': [1, 1, 1, 1, 0],
                'cp': [3, 0, 0, 3, 2],
                'trestbps': [145, 160, 120, 130, 130],
                'chol': [233, 286, 229, 250, 204],
                'fbs': [1, 0, 0, 0, 0],
                'restecg': [0, 0, 0, 1, 0],
                'thalach': [150, 108, 129, 187, 172],
                'exang': [0, 1, 1, 0, 0],
                'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
                'slope': [0, 1, 1, 0, 2],
                'ca': [0, 3, 2, 0, 0],
                'thal': [1, 2, 2, 2, 2],
                'target': [0, 1, 1, 0, 1]
            })
            
            # Initialize model with test data and then set master weights
            master_model = HeartDiseaseModel(test_data)
            master_model.set_weights(json.loads(result[0]))
            
            # Make predictions
            try:
                with st.spinner('Making predictions...'):
                    predictions = master_model.predict(df)
                    probabilities = master_model.predict_proba(df)
                    
                    # Create results DataFrame
                    results_df = df.copy()
                    results_df['Prediction'] = predictions
                    results_df['Probability'] = [prob[1] for prob in probabilities]
                    
                    # Display prediction results
                    st.subheader("Prediction Results")
                    st.write(results_df)
                    
                    # Download button for results
                    csv = results_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">Download Results CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Log the prediction action
                    log_user_action(
                        st.session_state.user_id,
                        "prediction",
                        f"Made predictions on {len(df)} samples using master model"
                    )
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your CSV file is properly formatted and contains all required columns.")

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