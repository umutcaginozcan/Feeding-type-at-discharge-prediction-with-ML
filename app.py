#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Breastfeeding Prediction - Flask Backend API
-------------------------------------------------
Simple Flask server that loads the trained Random Forest model
and provides a /predict endpoint for the web calculator.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Load model and metadata
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "trained_model.pkl"
METADATA_PATH = BASE_DIR / "feature_metadata.json"
INFO_PATH = BASE_DIR / "model_info.json"

print("Loading model and metadata...")
with open(MODEL_PATH, 'rb') as f:
    model_pipeline = pickle.load(f)

with open(METADATA_PATH, 'r') as f:
    feature_metadata = json.load(f)

with open(INFO_PATH, 'r') as f:
    model_info = json.load(f)

print("✓ Model loaded successfully!")
print(f"Model type: {model_info['model_type']}")
print(f"ROC-AUC: {model_info['performance_metrics']['roc_auc_macro']['mean']:.3f} ± {model_info['performance_metrics']['roc_auc_macro']['std']:.3f}")

@app.route('/')
def home():
    """Home page with API info"""
    return jsonify({
        'status': 'online',
        'model': model_info['model_type'],
        'training_date': model_info['training_date'],
        'n_features': model_info['n_features'],
        'performance': {
            'roc_auc': f"{model_info['performance_metrics']['roc_auc_macro']['mean']:.3f} ± {model_info['performance_metrics']['roc_auc_macro']['std']:.3f}",
            'accuracy': f"{model_info['performance_metrics']['accuracy']['mean']:.3f} ± {model_info['performance_metrics']['accuracy']['std']:.3f}"
        }
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Return the list of features needed for prediction"""
    return jsonify(feature_metadata)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict breastfeeding outcome based on input features.
    
    Expected input format:
    {
        "feature1": value1,
        "feature2": value2,
        ...
    }
    
    Returns:
    {
        "prediction": int (1, 2, or 3),
        "prediction_label": str,
        "probabilities": {
            "Exclusive Breastfeeding": float,
            "Formula Feeding": float,
            "Mixed Feeding": float
        },
        "confidence": float
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Create DataFrame with all expected features
        all_features = feature_metadata['num_features'] + feature_metadata['cat_features']
        
        # Initialize with None/NaN for missing features (will be imputed)
        feature_values = {}
        for feat in all_features:
            if feat in data:
                feature_values[feat] = data[feat]
            else:
                feature_values[feat] = np.nan
        
        # Create DataFrame
        input_df = pd.DataFrame([feature_values])
        
        # Make prediction
        prediction = model_pipeline.predict(input_df)[0]
        probabilities = model_pipeline.predict_proba(input_df)[0]
        
        # Map prediction to label
        # Model predicts 0, 1, 2 (0-indexed) but original class labels were 1, 2, 3
        class_labels_list = ['Exclusive Breastfeeding', 'Formula Feeding', 'Mixed Feeding']
        prediction_label = class_labels_list[prediction]
        
        # Calculate confidence (max probability)
        confidence = float(max(probabilities))
        
        # Format response
        response = {
            'prediction': int(prediction),
            'prediction_label': prediction_label,
            'probabilities': {
                'Exclusive Breastfeeding': float(probabilities[0]),
                'Formula Feeding': float(probabilities[1]),
                'Mixed Feeding': float(probabilities[2])
            },
            'confidence': confidence
        }
        
        return jsonify(response)
    
    except Exception as e:
        print("Error during prediction:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 NICU Breastfeeding Prediction API")
    print("=" * 60)
    print(f"Model: {model_info['model_type']}")
    print(f"Features: {model_info['n_features']}")
    print(f"Performance: ROC-AUC = {model_info['performance_metrics']['roc_auc_macro']['mean']:.3f}")
    print("=" * 60)
    print("\nStarting server on http://localhost:5000")
    print("Open calculator.html in your browser to use the tool")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, port=5000)
