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
import tensorflow as tf

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
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
        if data is not None:
            self._prepare_data(data)
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_data(self, data):
        """Prepare data for training"""
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Scale features
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = y.values
    
    def train_epoch(self):
        """Train the model for one epoch and return the loss"""
        if not hasattr(self, 'X_train'):
            raise ValueError("No training data available. Please initialize with data first.")
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        return history.history['loss'][0]
    
    def get_weights(self):
        """Get the model weights as a serializable format"""
        weights = []
        for layer in self.model.layers:
            layer_weights = []
            for w in layer.get_weights():
                layer_weights.append(w.tolist())
            weights.append(layer_weights)
        return weights
    
    def set_weights(self, weights):
        """Set the model weights from a serializable format"""
        for layer, layer_weights in zip(self.model.layers, weights):
            layer.set_weights([np.array(w) for w in layer_weights])
    
    def predict(self, data):
        """Make predictions using the model"""
        # Scale the input data
        X = self.scaler.transform(data)
        
        # Make prediction
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, data):
        """Get prediction probabilities"""
        # Scale the input data
        X = self.scaler.transform(data)
        
        # Get probabilities
        return self.model.predict(X)

    def evaluate(self, test_data):
        """Evaluate the model on test data."""
        if isinstance(test_data, str):
            test_data = pd.read_csv(test_data)
        
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Preprocess test data
        X_test_processed = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.predict(X_test_processed)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

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
        self.model.train_epoch()
        
        # Return updated parameters and number of training samples
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Evaluate model
        accuracy = self.model.evaluate(self.csv_path)
        loss = 1.0 - accuracy
        
        # Return loss, number of test samples, and metrics
        return float(loss), len(self.X_test), {"accuracy": accuracy} 