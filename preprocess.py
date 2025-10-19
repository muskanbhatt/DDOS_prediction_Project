
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load scaler once
def preprocess_input(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])  # Avoid using it for test
    df = pd.get_dummies(df)
    scaler = joblib.load('models/scaler.pkl')
    df_scaled = scaler.transform(df)
    return df_scaled