from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load models
model_path = "models/LogisticRegression.pkl"
scaler_path = "models/Normalization_model.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template("Maincc.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part in the request."
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file."
    
    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        
        # Drop label if present
        if 'Label' in df.columns:
            df = df.drop(columns=['Label'])
        
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        
        if df.empty:
            return "Uploaded file contains no valid data after cleaning."
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Predict
        predictions = model.predict(df_scaled)
        
        # Map prediction values to names for clarity
        pred_labels = ['Benign' if pred == 0 else 'DDOS' for pred in predictions]
        
        # Add prediction column to original DataFrame (unscaled)
        df['Prediction'] = pred_labels
        
        # Convert DataFrame to HTML table (safe to pass to template)
        table_html = df.to_html(classes='table table-striped', index=False)
        
        return render_template("result.html", table=table_html)

    return "Something went wrong."

if __name__ == "__main__":
    app.run(debug=True)
