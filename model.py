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
    model.classes_ = np.arange(13)  # Initially 13 classes
    model.coef_ = np.zeros((2, 13))  # Coefficient matrix
    model.intercept_ = np.zeros(2)    # Intercept vector
    return model

class HeartDiseaseModel:
    def __init__(self, csv_path=None):
        self.model = create_model()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        if csv_path:
            self.initialize_with_data(csv_path)

    def initialize_with_data(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit imputer and scaler
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        return self

    def preprocess_data(self, X):
        # Handle missing values and scale features
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def train(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Fit imputer and scaler if not already fitted
        if not hasattr(self.imputer, 'statistics_'):
            X_imputed = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            # Use existing imputer and scaler
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def evaluate(self, test_csv_path):
        test_data = pd.read_csv(test_csv_path)
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Preprocess test data
        X_test_processed = self.preprocess_data(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_processed)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def predict(self, X):
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        return self.model.predict(X_processed)

    def get_weights(self):
        # Convert numpy arrays to lists for JSON serialization
        return {
            'coef': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist()
        }

    def set_weights(self, weights):
        # Convert lists back to numpy arrays
        self.model.coef_ = np.array(weights['coef'])
        self.model.intercept_ = np.array(weights['intercept'])

    def save(self, path):
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
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