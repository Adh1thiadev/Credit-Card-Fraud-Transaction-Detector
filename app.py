from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ======================================================
# Load Model & Columns (same model as before)
# ======================================================
MODEL_PATH = 'model/model.pkl'
COLUMNS_PATH = 'model/columns.json'

model = None
model_columns = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(COLUMNS_PATH):
    try:
        with open(COLUMNS_PATH, 'r') as f:
            model_columns = json.load(f)
    except Exception:
        model_columns = None

os.makedirs('dataset', exist_ok=True)

# ======================================================
# HOME PAGE
# ======================================================
@app.route('/')
def home():
    return render_template('index.html')

# ======================================================
# MANUAL ENTRY PREDICTION
# ======================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)

        if model is None:
            raise RuntimeError("Model not found. Train model first.")

        prediction = model.predict(features_array)[0]
        result = "❌ Fraudulent Transaction" if prediction == 1 else "✅ Legit Transaction"
        return render_template('result.html', prediction_text=result)
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

# ======================================================
# BATCH UPLOAD PREDICTION
# ======================================================
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return render_template('batch_result.html', table="<p>No file uploaded</p>")
        if not allowed_file(file.filename):
            return render_template('batch_result.html', table="<p>Only CSV files allowed</p>")

        filename = secure_filename(file.filename)
        file_path = os.path.join('dataset', filename)
        file.save(file_path)

        data = pd.read_csv(file_path)
        if 'Class' in data.columns:
            data = data.drop(columns='Class')

        if model is None:
            raise RuntimeError("Model not loaded. Please train model first.")

        predictions = model.predict(data)
        data['Prediction'] = ['Fraudulent ❌' if p == 1 else 'Legit ✅' for p in predictions]

        # Save full batch output (for dashboard)
        batch_output_path = os.path.join('dataset', 'batch_output.csv')
        data.to_csv(batch_output_path, index=False)

        # Display top 50 for quick view
        table_html = data.head(50).to_html(classes='styled-table', index=False)
        return render_template('batch_result.html', table=table_html)
    except Exception as e:
        return render_template('batch_result.html', table=f"<p>Error: {str(e)}</p>")

# ======================================================
# DASHBOARD ROUTE
# ======================================================
@app.route('/dashboard')
def dashboard():
    batch_path = os.path.join('dataset', 'batch_output.csv')
    stats = {'total': 0, 'fraud': 0, 'legit': 0}
    recent = None

    # Read previous batch predictions
    if os.path.exists(batch_path):
        try:
            df = pd.read_csv(batch_path)
            stats['total'] = len(df)
            if 'Prediction' in df.columns:
                stats['fraud'] = int(df['Prediction'].str.contains('Fraud').sum())
                stats['legit'] = int(df['Prediction'].str.contains('Legit').sum())
            recent = df.tail(10).to_dict(orient='records')
        except Exception as e:
            print("Error reading batch_output.csv:", e)

    # Evaluation images (from train_model.py)
    images = {'confusion': None, 'roc': None, 'shap': None}
    if os.path.exists(os.path.join('static', 'images', 'confusion_matrix.png')):
        images['confusion'] = 'images/confusion_matrix.png'
    if os.path.exists(os.path.join('static', 'images', 'roc_curve.png')):
        images['roc'] = 'images/roc_curve.png'
    if os.path.exists(os.path.join('static', 'images', 'shap_summary.png')):
        images['shap'] = 'images/shap_summary.png'

    return render_template('dashboard.html', stats=stats, recent=recent, images=images)

# ======================================================
# DOWNLOAD LAST BATCH RESULT
# ======================================================
@app.route('/download/batch')
def download_batch():
    batch_path = os.path.join('dataset', 'batch_output.csv')
    if os.path.exists(batch_path):
        return send_from_directory('dataset', 'batch_output.csv', as_attachment=True)
    return redirect(url_for('home'))

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
