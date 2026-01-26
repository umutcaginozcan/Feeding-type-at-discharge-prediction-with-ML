# 🤱 NICU Breastfeeding Prediction Calculator

A machine learning-powered web application to predict feeding type at discharge (Exclusive Breastfeeding, Formula Feeding, or Mixed Feeding) for NICU infants based on early feeding data and clinical parameters.

![Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.1+-blue)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)

## 📸 Screenshot

![NICU Breastfeeding Calculator](calculator_screenshot.png)
*Sample prediction showing 60.5% probability for Exclusive Breastfeeding*

---

## 📊 Model Performance

The prediction model is a **Random Forest Classifier** trained on 1,064 NICU patient records with the following performance metrics:

| Metric | Score (5-Fold CV) |
|:-------|:------------------|
| **ROC-AUC (Macro)** | **0.868 ± 0.017** |
| **PR-AUC (Macro)** | 0.657 ± 0.044 |
| **Accuracy** | 0.792 ± 0.028 |
| **Balanced Accuracy** | 0.612 ± 0.046 |
| **F1-Score (Weighted)** | 0.790 ± 0.025 |

### Model Details
- **Algorithm**: Random Forest with SMOTE oversampling
- **Features**: 30 clinical and feeding parameters
- **Classes**: 3 (Exclusive Breastfeeding, Formula Feeding, Mixed Feeding)
- **Hyperparameters**: 449 estimators, max depth 25, balanced class weights
- **Training Date**: 2026-01-26

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/NICU-Breastfeeding-Paper.git
cd NICU-Breastfeeding-Paper
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Flask server**
```bash
python app.py
```

4. **Open the calculator**
   - The server will start on `http://localhost:5000`
   - Open `calculator.html` in your web browser
   - Or visit `http://localhost:5000` directly

---

## 💻 How to Use the Calculator

### Step 1: Enter Patient Information
Fill in the required fields (marked with *):
- **Birth Weight** (grams)
- **Gestational Age** (weeks)
- **Maternal Age** (years)

### Step 2: Enter Feeding Data
Provide feeding volumes for each day:
- **Day 1**: Breast milk and formula amounts
- **Day 2**: Breast milk and formula amounts
- **Day 3**: Breast milk and formula amounts

The calculator will automatically compute:
- Daily total volumes
- Breast milk ratios
- Volume changes between days
- Lactation momentum index
- Weight-adjusted feeding metrics

### Step 3: Optional Information
Add additional clinical data if available:
- Weight measurements (Day 1, 2, 3)
- COVID-19 period indicator
- Feeding method indicators
- Maternal education status
- Birth spacing
- And more...

### Step 4: Get Prediction
Click the **"🔮 Predict Feeding Outcome"** button to receive:
- Predicted feeding type at discharge
- Confidence level (percentage)
- Probability distribution across all three classes
- Visual bar chart representation

---

## 📁 Project Structure

```
NICU-Breastfeeding-Paper/
├── app.py                      # Flask API server
├── calculator.html             # Web interface
├── trained_model.pkl           # Trained Random Forest model
├── feature_metadata.json       # Feature definitions
├── model_info.json            # Model performance metrics
├── requirements.txt           # Python dependencies
├── Code/
│   ├── Stage-1-*.py           # Data preprocessing pipeline
│   ├── Stage-2-*.py           # Feature engineering
│   ├── Stage-3-*.py           # Model training & optimization
│   ├── Stage-4-*.py           # Model evaluation
│   ├── Stage-5-*.py           # Advanced analysis
│   └── Stage-6-export-model.py # Model export for deployment
└── excels-NICU-breatsfeeding-data/
    └── nicu_stage0_5_cleaned.xlsx  # Cleaned dataset
```

---

## 🔧 Technical Details

### Backend (Flask API)
- **Framework**: Flask 3.1+
- **Model Loading**: Pickle (scikit-learn pipeline)
- **CORS**: Enabled for local development
- **Endpoints**:
  - `GET /` - API status and info
  - `POST /predict` - Prediction endpoint
  - `GET /health` - Health check

### Frontend (HTML/CSS/JavaScript)
- **Design**: Modern gradient UI with glassmorphism
- **Validation**: Real-time form validation
- **Calculations**: Automatic feature engineering in browser
- **Visualization**: Probability bar charts

### Machine Learning Pipeline
1. **Preprocessing**: Median imputation for numeric features
2. **Resampling**: SMOTE for class balance
3. **Classification**: Random Forest with optimized hyperparameters
4. **Output**: Multi-class probabilities

---

## 📊 Features Used by the Model

The model uses 30 carefully engineered features including:

### Clinical Parameters
- Birth weight and gestational age
- Maternal age
- Daily weight measurements
- COVID-19 period indicator

### Feeding Metrics
- Daily breast milk volumes (Days 1-3)
- Daily formula volumes (Days 1-3)
- Total daily feeding volumes
- Breast milk ratios per day

### Engineered Features
- **Lactation Momentum**: Rate of breast milk increase
- **Resilience Index**: Feeding volume stability
- **Delta Volumes**: Day-to-day volume changes
- **Weight-adjusted**: Feeding per week of gestation

---

## 🧪 Testing

To test the prediction API:

```python
import requests
import json

# Sample test data
test_data = {
    "dogumagirligi(gram)": 3200,
    "gebelikhaftası": 38,
    "anneyasi": 28,
    "aldığıannesütü_ilkgün": 10,
    "aldığımamamiktari1.gün": 5,
    "beslenme2.gunannesutucc": 25,
    "beslenmemamamiktarı2.guncc": 5,
    "aldığıannesütü3.gun": 40,
    "aldığımamamiktari3.gun": 0
}

response = requests.post(
    "http://localhost:5000/predict",
    json=test_data,
    headers={"Content-Type": "application/json"}
)

print(json.dumps(response.json(), indent=2))
```

Expected output:
```json
{
  "prediction": "Exclusive Breastfeeding",
  "confidence": 60.5,
  "probabilities": {
    "Exclusive Breastfeeding": 60.5,
    "Formula Feeding": 36.2,
    "Mixed Feeding": 3.3
  }
}
```

---

## 📦 Dependencies

### Python Packages
```
Flask>=3.1.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
openpyxl>=3.1.0
flask-cors>=4.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔒 Privacy & Ethics

- **No PHI Storage**: Patient data is processed in-memory only
- **Local Processing**: All predictions happen locally, no external API calls
- **Research Use**: This tool is for research purposes only
- **Clinical Validation**: Not approved for clinical decision-making without physician oversight

---

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@software{nicu_breastfeeding_calculator,
  title={NICU Breastfeeding Prediction Calculator},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/NICU-Breastfeeding-Paper}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Your Name** - Initial work and model development

---

## 🙏 Acknowledgments

- NICU medical staff for data collection
- Research team for clinical expertise
- Open-source ML community for tools and libraries

---

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub Issues: [Report a bug](https://github.com/yourusername/NICU-Breastfeeding-Paper/issues)

---

## 🔄 Version History

- **v1.0.0** (2026-01-26)
  - Initial release
  - Random Forest model with 86.8% ROC-AUC
  - Web-based calculator interface
  - Flask API deployment

---

**Built with ❤️ for improving NICU patient outcomes**
