import os
import flwr as fl
import numpy as np
from pathlib import Path
from model import HeartDiseaseModel
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
AUTH_TOKEN = os.getenv("FLWR_AUTH_TOKEN")

def load_data(data_path):
    """Load and preprocess data from CSV file."""
    data = pd.read_csv(data_path)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

class MedicalClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = HeartDiseaseModel()
        
        # Load training and validation data
        self.X_train, self.y_train = load_data(DATA_DIR / "train.csv")
        self.X_val, self.y_val = load_data(DATA_DIR / "val.csv")
        
        # Initialize model with training data
        self.model.initialize_with_data(pd.concat([self.X_train, pd.Series(self.y_train, name='target')], axis=1))

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        weights = self.model.get_weights()
        return [
            np.array(weights['coef']),
            np.array(weights['intercept'])
        ]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        self.model.set_weights({
            'coef': parameters[0],
            'intercept': parameters[1]
        })

    def fit(self, parameters, config):
        """Train model on local data."""
        self.set_parameters(parameters)
        
        # Train model
        train_data = pd.concat([self.X_train, pd.Series(self.y_train, name='target')], axis=1)
        self.model.initialize_with_data(train_data)
        
        # Get updated parameters
        updated_params = self.get_parameters(config)
        
        return updated_params, len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local validation data."""
        self.set_parameters(parameters)
        
        # Evaluate model
        val_data = pd.concat([self.X_val, pd.Series(self.y_val, name='target')], axis=1)
        accuracy = self.model.evaluate(val_data)
        loss = 1.0 - accuracy
        
        return float(loss), len(self.X_val), {"accuracy": float(accuracy)}

def main():
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=MedicalClient(),
        root_certificates=Path("ca.pem") if os.path.exists("ca.pem") else None,
        transport_credentials=fl.common.grpc_credentials(
            root_certificates=Path("ca.pem") if os.path.exists("ca.pem") else None
        ) if os.path.exists("ca.pem") else None
    )

if __name__ == "__main__":
    main() 