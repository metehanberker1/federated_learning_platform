import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import flwr as fl

def create_model():
    model = LogisticRegression(
        C=0.01,  # Strong regularization
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000
    )
    model.classes_ = np.array([0, 1])  # Binary classification
    model.coef_ = np.zeros((1, 13))  # Coefficient matrix for 13 features
    model.intercept_ = np.zeros(1)    # Intercept vector
    return model

class HeartDiseaseModel:
    def __init__(self, data=None):
        self.model = create_model()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        if data is not None:
            self.initialize_with_data(data)

    def initialize_with_data(self, data):
        """Initialize the model with either a DataFrame or a path to a CSV file."""
        if isinstance(data, str):
            # If data is a string, assume it's a file path
            data = pd.read_csv(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("data must be either a DataFrame or a path to a CSV file")
        
        # Ensure the DataFrame has the required columns
        required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        if 'target' in data.columns:
            X = data[required_columns]
            y = data['target']
        else:
            X = data[required_columns]
            y = None  # For prediction only
        
        # Fit imputer and scaler
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        if y is not None:
            # Train model if target is available
            self.model.fit(X_scaled, y)
        
        return self

    def preprocess_data(self, X):
        """Preprocess the input data using fitted imputer and scaler."""
        if not hasattr(self.imputer, 'statistics_') or not hasattr(self.scaler, 'mean_'):
            raise ValueError("Model has not been initialized with data. Call initialize_with_data first.")
        
        # Handle missing values and scale features
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def predict(self, X):
        """Make predictions on new data."""
        # Ensure X has the correct columns
        required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        if not all(col in X.columns for col in required_columns):
            raise ValueError(f"Input data must contain all required columns: {required_columns}")
        
        # Preprocess data
        X = X[required_columns]  # Ensure correct column order
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """Get probability estimates for predictions."""
        # Ensure X has the correct columns
        required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        if not all(col in X.columns for col in required_columns):
            raise ValueError(f"Input data must contain all required columns: {required_columns}")
        
        # Preprocess data
        X = X[required_columns]  # Ensure correct column order
        X_processed = self.preprocess_data(X)
        
        # Get probability estimates
        return self.model.predict_proba(X_processed)

    def evaluate(self, test_data):
        """Evaluate the model on test data."""
        if isinstance(test_data, str):
            test_data = pd.read_csv(test_data)
        
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Preprocess test data
        X_test_processed = self.preprocess_data(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_processed)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def get_weights(self):
        """Get model weights for federated learning."""
        return {
            'coef': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist()
        }

    def set_weights(self, weights):
        """Set model weights for federated learning."""
        self.model.coef_ = np.array(weights['coef'])
        self.model.intercept_ = np.array(weights['intercept'])

    def save(self, path):
        """Save the model to a file."""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        """Load the model from a file."""
        return joblib.load(path)

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id, csv_path):
        self.client_id = client_id
        self.csv_path = csv_path
        self.model = HeartDiseaseModel(csv_path)  # Initialize with data to fit imputer and scaler
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_split_data(csv_path)

    def load_and_split_data(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Train model
        self.model.train(self.csv_path)
        
        # Return updated parameters and number of training samples
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Evaluate model
        accuracy = self.model.evaluate(self.csv_path)
        loss = 1.0 - accuracy
        
        # Return loss, number of test samples, and metrics
        return float(loss), len(self.X_test), {"accuracy": accuracy} 