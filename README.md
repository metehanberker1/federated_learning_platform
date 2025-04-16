# Medical Analysis App

A Streamlit-based web application for heart disease prediction using federated learning.

## Features

- User authentication and profile management
- Data upload and model initialization
- Heart disease prediction with probability scores
- User action logging
- Modern and responsive UI

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run streamlit_app.py
```

## Deploying to Streamlit Cloud

1. Create a Streamlit Cloud account at https://streamlit.io/cloud
2. Connect your GitHub repository
3. Deploy the app by selecting the repository and branch
4. The app will be automatically deployed and available at a public URL

## Data Format

The application expects a CSV file with the following columns:
- age: Age in years
- sex: Sex (1 = male; 0 = female)
- cp: Chest pain type (0-3)
- trestbps: Resting blood pressure
- chol: Serum cholesterol in mg/dl
- fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- restecg: Resting electrocardiographic results (0-2)
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slope: Slope of the peak exercise ST segment (0-2)
- ca: Number of major vessels (0-3) colored by fluoroscopy
- thal: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
- target: Heart disease diagnosis (1 = present; 0 = absent)

## Security Note

This application uses SQLite for demonstration purposes. For production deployment, consider:
1. Using a more robust database system (e.g., PostgreSQL)
2. Implementing proper security measures
3. Setting up environment variables for sensitive data
4. Adding rate limiting and other security features 