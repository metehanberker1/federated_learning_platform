from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from model import HeartDiseaseModel, HospitalClient
import flwr as fl
from flwr.server.strategy import FedAvg
import threading
import time
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
from flwr.common import secagg_mod
from flwr.server.client_manager import SimpleClientManager
from flwr.common import Parameters
from flwr.server.client_proxy import ClientProxy
import jwt
from pathlib import Path
from dotenv import load_dotenv

app = Flask(__name__)

# Initialize the global model with test data
global_model = HeartDiseaseModel('data/test_heart_disease.csv')
client_weights = []

# Load environment variables
load_dotenv()

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "10"))
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")

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

# Initialize database
init_db()

@app.route('/verify_user', methods=['POST'])
def verify_user():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not all([username, password]):
            return jsonify({'error': 'Missing credentials'}), 400
        
        conn = sqlite3.connect('federated_learning.db')
        c = conn.cursor()
        
        c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        conn.close()
        
        if not result or not check_password_hash(result[1], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        return jsonify({'message': 'Login successful', 'user_id': result[0]}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create_user', methods=['POST'])
def create_user():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        conn = sqlite3.connect('federated_learning.db')
        c = conn.cursor()
        
        # Check if username or email already exists
        c.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if c.fetchone():
            return jsonify({'error': 'Username or email already exists'}), 409
        
        # Create new user
        password_hash = generate_password_hash(password)
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                 (username, email, password_hash))
        
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        
        return jsonify({'message': 'User created successfully', 'user_id': user_id}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_user_profile/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    try:
        conn = sqlite3.connect('federated_learning.db')
        c = conn.cursor()
        
        c.execute('SELECT id, username, email, created_at FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'created_at': user[3]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_profile/<int:user_id>', methods=['PUT'])
def update_profile(user_id):
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'No data to update'}), 400
        
        conn = sqlite3.connect('federated_learning.db')
        c = conn.cursor()
        
        c.execute('UPDATE users SET email = ? WHERE id = ?', (email, user_id))
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Profile updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_user_actions/<int:user_id>', methods=['GET'])
def get_user_actions(user_id):
    try:
        conn = sqlite3.connect('federated_learning.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT action_type, details, timestamp 
            FROM user_actions 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
        ''', (user_id,))
        
        actions = c.fetchall()
        conn.close()
        
        return jsonify([{
            'action_type': action[0],
            'details': action[1],
            'timestamp': action[2]
        } for action in actions]), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        print(f"Error logging user action: {str(e)}")

class AuthClientManager(SimpleClientManager):
    """Custom client manager that verifies JWT tokens."""
    
    def register(self, client: ClientProxy) -> bool:
        """Verify client's JWT token before registration."""
        try:
            # Get token from client's metadata
            metadata = client.get_properties()
            token = metadata.get("token")
            
            if not token:
                print("No token provided")
                return False
            
            # Verify token
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            
            # If verification succeeds, register client
            return super().register(client)
            
        except jwt.InvalidTokenError:
            print("Invalid token")
            return False
        except Exception as e:
            print(f"Error during registration: {e}")
            return False

def get_on_fit_config(rnd: int) -> Dict[str, str]:
    """Return training configuration dict for each round."""
    return {
        "round": str(rnd),
        "epochs": "1",
        "batch_size": "32"
    }

def get_evaluate_fn(model_path: str):
    """Return an evaluation function for server-side evaluation."""
    
    def evaluate(
        server_round: int,
        parameters: Parameters,
        config: Dict[str, str],
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Evaluate model parameters using validation set."""
        # Load validation data and model
        # Implement if needed for centralized evaluation
        return None  # No centralized evaluation for now
    
    return evaluate

def start_flask_server():
    app.run(host='0.0.0.0', port=5000)

@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    try:
        data = request.get_json()
        weights = data.get('weights')
        user_id = data.get('user_id')
        
        if weights and user_id:
            client_weights.append(weights)
            log_user_action(user_id, 'submit_weights', 'Submitted local model weights')
            return jsonify({'message': 'Weights submitted successfully'}), 200
        return jsonify({'error': 'No weights provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
            
        if client_weights:
            # Average the weights
            avg_weights = np.mean(client_weights, axis=0)
            global_model.set_weights(avg_weights)
            
            # Retrain with test data
            global_model.initialize_with_data('data/test_heart_disease.csv')
            
            log_user_action(user_id, 'get_global_model', 'Retrieved global model')
            return jsonify({'message': 'Global model updated successfully'}), 200
        return jsonify({'message': 'No client weights available'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_global_model', methods=['GET'])
def evaluate_global():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
            
        accuracy = global_model.evaluate('data/test_heart_disease.csv')
        log_user_action(user_id, 'evaluate_model', f'Model accuracy: {accuracy}')
        return jsonify({'accuracy': accuracy}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        features = data.get('features')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
            
        if not features:
            return jsonify({'error': 'No feature data provided'}), 400
            
        # Convert features to DataFrame
        df = pd.DataFrame(features)
        
        # Make predictions
        predictions = global_model.predict(df)
        
        # Convert predictions to human-readable messages
        results = []
        for pred in predictions:
            if pred == 1:
                results.append("Patient is predicted to have heart disease")
            else:
                results.append("Patient is predicted to be healthy")
        
        # Log the prediction action
        log_user_action(user_id, 'predict', f'Made predictions for {len(predictions)} patients')
        
        return jsonify({
            "predictions": predictions.tolist(),
            "messages": results
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    # Initialize strategy
    strategy = FedAvg(
        min_available_clients=MIN_CLIENTS,
        evaluate_fn=get_evaluate_fn("global_model.pkl"),
        on_fit_config_fn=get_on_fit_config,
        mods=[secagg_mod()]  # Enable secure aggregation
    )
    
    # SSL/TLS certificates
    cert_dir = Path("certificates")
    ssl_certificates = None
    if cert_dir.exists():
        ssl_certificates = (
            str(cert_dir / "server.pem"),  # Server certificate
            str(cert_dir / "server.key"),  # Server private key
            str(cert_dir / "ca.pem")       # Root certificate
        )
    
    # Start Flower server
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=AuthClientManager(),
        certificates=ssl_certificates
    )

def create_client_token(client_id: str, expiration_days: int = 30) -> str:
    """Create a JWT token for client authentication."""
    expiration = datetime.utcnow() + timedelta(days=expiration_days)
    payload = {
        "client_id": client_id,
        "exp": expiration
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

if __name__ == '__main__':
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start Flower server in the main thread
    main() 