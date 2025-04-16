import requests
import json

# Server URL
SERVER_URL = "http://localhost:5000"

def test_prediction():
    # First, let's create a test user (if you don't already have one)
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    }
    
    # Create user
    create_response = requests.post(f"{SERVER_URL}/create_user", json=user_data)
    if create_response.status_code == 201:
        user_id = create_response.json()['user_id']
        print(f"Created test user with ID: {user_id}")
    elif create_response.status_code == 409:
        # User exists, let's login
        login_response = requests.post(f"{SERVER_URL}/verify_user", 
                                     json={"username": "testuser", "password": "testpass123"})
        if login_response.status_code == 200:
            user_id = login_response.json()['user_id']
            print(f"Logged in with user ID: {user_id}")
        else:
            print("Failed to login:", login_response.json())
            return
    else:
        print("Failed to create user:", create_response.json())
        return

    # Sample data for prediction (one patient)
    prediction_data = {
        "user_id": user_id,
        "features": [{
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }]
    }

    # Make prediction request
    try:
        response = requests.post(f"{SERVER_URL}/predict", json=prediction_data)
        
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print("Numerical predictions:", result["predictions"])
            print("Messages:", result["messages"])
        else:
            print("Error making prediction:", response.json())
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_prediction() 