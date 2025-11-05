# 💳 Credit Card Fraud Detection System

A machine learning–powered Flask web application that detects fraudulent credit card transactions in real-time.  
The system provides both **manual prediction** and **batch CSV upload**, along with **confidence score visualization**, **auto-logging**, and an **analytics dashboard** for recent predictions.

---

## 🚀 Features

✅ **Real-Time Prediction**
- Enter 30 transaction features manually and instantly predict whether a transaction is **Legit** or **Fraudulent**.  
- Displays model confidence as a colored progress bar.

✅ **Batch Upload Mode**
- Upload a CSV file of transactions for bulk fraud detection.  
- Each transaction is analyzed, and results (Fraud/Legit + Confidence) are displayed in a formatted table.  

✅ **Prediction Confidence**
- Displays fraud probability (%) for every prediction using model probabilities.  
- Dynamic progress bar visualization (green → orange → red).  

✅ **Auto Logging**
- Every manual or batch prediction is logged automatically to `dataset/predictions_log.csv` with timestamp, prediction, confidence, and short feature preview.  
- Supports dashboard analytics for reviewing recent activity.

✅ **Dashboard Analytics**
- Shows summary statistics and **recent predictions table** with fraud probabilities and modes (manual/batch).  
- Ready to extend with SHAP charts, model metrics, or fraud trend graphs.

✅ **Modern Dark UI**
- Classic navy-grey background, glassmorphic cards, and cyan-blue accent theme.  
- Two-column manual input grid for compact, professional layout.  
- Animated buttons, modern typography, and responsive design.

---

## 🧠 Machine Learning Model

The backend model is trained on the popular **Credit Card Fraud Detection Dataset** (`creditcard.csv`).

- **Algorithm:** Random Forest Classifier  
- **Balanced Training:** Dataset is undersampled to balance fraud vs. legit classes.  
- **Evaluation:** Accuracy, F1-Score, Confusion Matrix, and Classification Report are displayed in the training script output.  
- **Serialization:** Trained model saved as `model/model.pkl` using `joblib`.


CreditCardFraudDetection/
│
├── app.py                     # Flask web app (routes + prediction logic)
├── train_model.py             # Model training & evaluation
│
├── model/
│   └── model.pkl              # Saved RandomForest model
│
├── dataset/
│   ├── creditcard.csv         # Training dataset
│   ├── batch_output.csv       # Latest batch results
│   └── predictions_log.csv    # Auto-logged predictions
│
├── templates/
│   ├── index.html             # Home page (manual & batch input)
│   ├── result.html            # Result page with confidence bar
│   ├── batch_result.html      # Batch results table
│   └── dashboard.html         # Analytics dashboard
│
└── static/
    └── css/
        └── style.css          # Modern dark-glass UI styling

```bash
python train_model.py
