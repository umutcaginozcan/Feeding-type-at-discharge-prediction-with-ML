#!/usr/bin/env python3
"""
Direct test of the model without using the Flask API
"""
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load model and metadata
BASE_DIR = Path("/Users/umutcaginozcan/Desktop/NICU Breastfeeding Paper")
MODEL_PATH = BASE_DIR / "trained_model.pkl"
METADATA_PATH = BASE_DIR / "feature_metadata.json"

print("Loading model...")
with open(MODEL_PATH, 'rb') as f:
    model_pipeline = pickle.load(f)

with open(METADATA_PATH, 'r') as f:
    feature_metadata = json.load(f)

print("✓ Model loaded!\n")

# Test Case 1: Favorable for Exclusive Breastfeeding
case1 = {
    "dogumagirligi(gram)": 3200,
    "gebelikhaftası": 38,
    "anneyasi": 28,
    "aldığımamamiktari1.gün": 5,
    "aldığıannesütü_ilkgün": 10,
    "beslenme2.gunannesutucc": 25,
    "beslenmemamamiktarı2.guncc": 5,
    "beslenmetotali2.gün": 30,
    "aldığıannesütü3.gun": 40,
    "aldığımamamiktari3.gun": 0,
    "beslenmetotali3.gun": 40,
    "verilisyolu3gun": 4,
    "annesutuemzirmeegitimidurumu": 1,
    "covid19sonrasi": 1,
    "bebek_dostu_20temmuz2018": 1,
    "ilk_gün_anne_sütü_1111": 1,
    "ilk_gün_emzirme_111": 1
}

# Test Case 2: Likely Formula Feeding
case2 = {
    "dogumagirligi(gram)": 1800,
    "gebelikhaftası": 32,
    "anneyasi": 35,
    "aldığımamamiktari1.gün": 15,
    "aldığıannesütü_ilkgün": 2,
    "beslenme2.gunannesutucc": 5,
    "beslenmemamamiktarı2.guncc": 20,
    "beslenmetotali2.gün": 25,
    "aldığıannesütü3.gun": 8,
    "aldığımamamiktari3.gun": 25,
    "beslenmetotali3.gun": 33,
    "annesutuemzirmeegitimidurumu": 0,
    "covid19sonrasi": 0,
    "ilk_gün_anne_sütü_1111": 0
}

# Test Case 3: Mixed Feeding
case3 = {
    "dogumagirligi(gram)": 2800,
    "gebelikhaftası": 36,
    "anneyasi": 30,
    "aldığımamamiktari1.gün": 10,
    "aldığıannesütü_ilkgün": 8,
    "beslenme2.gunannesutucc": 15,
    "beslenmemamamiktarı2.guncc": 15,
    "beslenmetotali2.gün": 30,
    "aldığıannesütü3.gun": 20,
    "aldığımamamiktari3.gun": 15,
    "beslenmetotali3.gun": 35,
    "annesutuemzirmeegitimidurumu": 1,
    "covid19sonrasi": 1,
    "ilk_gün_anne_sütü_1111": 1
}

test_cases = [
    ("Case 1: Favorable for Exclusive Breastfeeding", case1),
    ("Case 2: Preterm, Likely Formula Feeding", case2),
    ("Case 3: Balanced, Likely Mixed Feeding", case3)
]

print("=" * 80)
print("🏥 NICU BREASTFEEDING PREDICTION - DIRECT MODEL TEST")
print("=" * 80)
print()

for case_name, case_data in test_cases:
    print(f"📋 {case_name}")
    print("-" * 80)
    print(f"   Birth Weight: {case_data.get('dogumagirligi(gram)', 'N/A')}g")
    print(f"   Gestational Age: {case_data.get('gebelikhaftası', 'N/A')} weeks")
    print(f"   Maternal Age: {case_data.get('anneyasi', 'N/A')} years")
    print(f"   Day 1: BM={case_data.get('aldığıannesütü_ilkgün', 0)}mL, Formula={case_data.get('aldığımamamiktari1.gün', 0)}mL")
    print(f"   Day 2: BM={case_data.get('beslenme2.gunannesutucc', 0)}mL, Formula={case_data.get('beslenmemamamiktarı2.guncc', 0)}mL")
    print(f"   Day 3: BM={case_data.get('aldığıannesütü3.gun', 0)}mL, Formula={case_data.get('aldığımamamiktari3.gun', 0)}mL")
    print(f"   BF Education: {'Yes' if case_data.get('annesutuemzirmeegitimidurumu', 0) == 1 else 'No'}")
    
    # Prepare data
    all_features = feature_metadata['num_features'] + feature_metadata['cat_features']
    feature_values = {feat: case_data.get(feat, np.nan) for feat in all_features}
    input_df = pd.DataFrame([feature_values])
    
    # Make prediction
    prediction = model_pipeline.predict(input_df)[0]
    probabilities = model_pipeline.predict_proba(input_df)[0]
    
    # Map to label
    # Model predicts 0, 1, 2 but original labels were 1, 2, 3
    class_labels_list = ['Exclusive Breastfeeding', 'Formula Feeding', 'Mixed Feeding']
    prediction_label = class_labels_list[prediction]
    confidence = max(probabilities)
    
    # Display results
    print(f"\n   {'🔮 PREDICTION':20s}: {prediction_label}")
    print(f"   {'✅ Confidence':20s}: {confidence*100:.1f}%")
    print(f"\n   📊 Probabilities:")
    
    outcomes = ['Exclusive Breastfeeding', 'Formula Feeding', 'Mixed Feeding']
    for i, outcome in enumerate(outcomes):
        prob = probabilities[i]
        bar_length = int(prob * 50)
        bar = "█" * bar_length
        emoji = "🤱" if i == 0 else "🍼" if i == 1 else "🤝"
        print(f"      {emoji} {outcome:28s}: {prob*100:5.1f}% │{bar}")
    
    print()
    print()

print("=" * 80)
print("✅ All predictions completed successfully!")
print("=" * 80)
