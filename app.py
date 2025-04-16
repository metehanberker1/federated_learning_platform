import streamlit as st
import pandas as pd
import os
from datetime import datetime
import requests
from model import HeartDiseaseModel

# Set page configuration
st.set_page_config(
    page_title="Medical Analysis App",
    page_icon="🏥",
    layout="wide"
)

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'local_model' not in st.session_state:
    st.session_state.local_model = None

# Server URL
SERVER_URL = "http://localhost:5000"

def login_page():
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            response = requests.post(
                f"{SERVER_URL}/verify_user",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_id = data['user_id']
                st.rerun()  # This will trigger a complete page rerun
            else:
                st.error("Invalid username or password")
    
    if not st.session_state.authenticated:  # Only show register option if not authenticated
        st.write("Don't have an account?")
        if st.button("Register"):
            st.session_state.show_register = True

def register_page():
    st.title("Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                response = requests.post(
                    f"{SERVER_URL}/create_user",
                    json={
                        "username": username,
                        "email": email,
                        "password": password
                    }
                )
                
                if response.status_code == 201:
                    st.success("Registration successful! Please login.")
                    st.session_state.show_register = False
                elif response.status_code == 409:
                    st.error("Username or email already exists!")
                else:
                    st.error("Registration failed. Please try again.")

def profile_page():
    st.title("Profile")
    
    if st.session_state.username:
        try:
            response = requests.get(
                f"{SERVER_URL}/get_user_profile/{st.session_state.user_id}"
            )
            
            if response.status_code == 200:
                profile = response.json()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Profile Information")
                    st.write(f"Username: {profile['username']}")
                    st.write(f"Email: {profile['email']}")
                    st.write(f"Member since: {profile['created_at']}")
                
                with col2:
                    st.subheader("Action History")
                    actions_response = requests.get(
                        f"{SERVER_URL}/get_user_actions/{st.session_state.user_id}"
                    )
                    
                    if actions_response.status_code == 200:
                        actions = actions_response.json()
                        for action in actions:
                            st.write(f"**{action['action_type']}** - {action['timestamp']}")
                            if action['details']:
                                st.write(f"Details: {action['details']}")
                            st.write("---")
                    else:
                        st.error("Failed to load action history")
            else:
                st.error("Failed to load profile")
                
        except Exception as e:
            st.error(f"Error loading profile: {str(e)}")

def document_upload_page():
    st.title("Document Upload")
    
    if not st.session_state.authenticated:
        st.warning("Please login first to upload documents.")
        return
    
    st.write("### Upload your dataset")
    st.write("Upload your data file to train a local model. The file should be in CSV format with the target variable.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Save the file
        file_path = os.path.join('uploads', f"{st.session_state.user_id}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("Train Model"):
            try:
                # Initialize and train model
                model = HeartDiseaseModel()
                model.train(file_path)
                st.session_state.local_model = model
                
                # Evaluate local model
                accuracy = model.evaluate('data/test_heart_disease.csv')
                st.success(f"Local model trained successfully! Test accuracy: {accuracy * 100:.2f}%")
                
                # Submit weights to server
                weights = model.get_weights()
                response = requests.post(
                    f"{SERVER_URL}/submit_weights",
                    json={
                        'weights': weights,
                        'user_id': st.session_state.user_id
                    }
                )
                
                if response.status_code == 200:
                    st.success("Model weights submitted to server successfully!")
                else:
                    st.error("Failed to submit weights to server")
                    
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def inference_page():
    st.title("Make Predictions")
    
    if not st.session_state.authenticated:
        st.warning("Please login first to use the inference service.")
        return
    
    st.write("### Upload data for prediction")
    st.write("Upload a CSV file containing patient data (without the target variable) to get predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(data.head())
            
            if st.button("Get Predictions"):
                prediction_data = {
                    "user_id": st.session_state.user_id,
                    "features": data.to_dict('records')
                }
                
                response = requests.post(f"{SERVER_URL}/predict", json=prediction_data)
                
                if response.status_code == 200:
                    result = response.json()
                    results_df = pd.DataFrame({
                        'Prediction': result["predictions"],
                        'Interpretation': result["messages"]
                    })
                    
                    st.write("### Prediction Results")
                    st.dataframe(results_df)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
                else:
                    st.error(f"Error making prediction: {response.json().get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def main():
    st.sidebar.title("Navigation")
    
    if st.session_state.authenticated:
        st.sidebar.write(f"Welcome, {st.session_state.username}!")
        
        page = st.sidebar.radio(
            "Select Page",
            ["Profile", "Document Upload", "Make Predictions"]
        )
        
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_id = None
        
        if page == "Profile":
            profile_page()
        elif page == "Document Upload":
            document_upload_page()
        elif page == "Make Predictions":
            inference_page()
    else:
        st.sidebar.write("Please login to access the app")
        if hasattr(st.session_state, 'show_register') and st.session_state.show_register:
            register_page()
        else:
            login_page()

if __name__ == "__main__":
    main() 