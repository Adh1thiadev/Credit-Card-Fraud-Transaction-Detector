# ==========================================================
# Credit Card Fraud Detection - Model Training Script
# (Enhanced: saves columns.json and evaluation plots)
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import json
import matplotlib.pyplot as plt

# Optional SHAP
USE_SHAP = True
try:
    import shap
except Exception:
    USE_SHAP = False
    print('SHAP not installed — skipping SHAP explainability plots')

# ==========================================================
# Step 1: Load Dataset
# ==========================================================
data_path = 'dataset/creditcard.csv'  # update path if needed
print("📂 Loading dataset...")
df = pd.read_csv(data_path)
print(f"✅ Dataset loaded successfully! Shape: {df.shape}")

# ==========================================================
# Step 2: Explore and Balance Dataset
# ==========================================================
print("\nClass distribution before balancing:")
print(df['Class'].value_counts())

legit = df[df.Class == 0]
fraud = df[df.Class == 1]

# Undersample to get balanced dataset
legit_sample = legit.sample(n=len(fraud), random_state=42)
balanced_df = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

print("\n✅ Balanced dataset created:")
print(balanced_df['Class'].value_counts())

# ==========================================================
# Step 3: Split into Features (X) and Labels (Y)
# ==========================================================
X = balanced_df.drop(columns='Class', axis=1)
Y = balanced_df['Class']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

print(f"\nTraining size: {X_train.shape}, Test size: {X_test.shape}")

# ==========================================================
# Step 4: Build and Train the Model
# ==========================================================
# You can switch between LogisticRegression and RandomForest here
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=10,
        class_weight='balanced'
    )
)

print("\n🚀 Training model...")
model.fit(X_train, Y_train)
print("✅ Model training complete!")

# ==========================================================
# Step 5: Evaluate the Model
# ==========================================================
Y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)

print("\n📊 Model Evaluation:")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred, digits=4))

# Also compute and print ROC AUC if possible
try:
    from sklearn.metrics import roc_auc_score
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(Y_test, y_proba)
    print(f"\nROC AUC: {roc:.4f}")
except Exception:
    print('\nCould not compute ROC AUC (predict_proba unavailable)')

# ==========================================================
# Step 6: Save the Model and columns
# ==========================================================
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
print("\n💾 Model saved successfully at: model/model.pkl")

# Save feature column order
columns = X.columns.tolist()
with open('model/columns.json', 'w') as f:
    json.dump(columns, f)
print('💾 Feature columns saved at model/columns.json')

# ==========================================================
# Step 7: Save evaluation images (confusion matrix, ROC)
# ==========================================================
os.makedirs('static/images', exist_ok=True)
try:
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    # Confusion matrix plot
    ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred)
    plt.title('Confusion Matrix')
    plt.savefig('static/images/confusion_matrix.png')
    plt.close()

    # ROC curve plot if probabilities are available
    if hasattr(model, 'predict_proba') or True:
        try:
            # pipeline supports predict_proba through underlying classifier
            y_score = model.predict_proba(X_test)[:, 1]
            RocCurveDisplay.from_predictions(Y_test, y_score)
            plt.title('ROC Curve')
            plt.savefig('static/images/roc_curve.png')
            plt.close()
        except Exception as e:
            print('Could not save ROC plot:', e)
    else:
        print('predict_proba not available; skipping ROC plot')
except Exception as e:
    print('Could not generate evaluation images:', e)

# SHAP summary plot (optional)
if USE_SHAP:
    try:
        # In pipeline, access the final estimator
        # For RandomForest inside a pipeline: model.named_steps['randomforestclassifier']
        final_estimator = None
        if hasattr(model, 'named_steps'):
            # find classifier step name
            for name, step in model.named_steps.items():
                # choose the first estimator with predict_proba
                if hasattr(step, 'predict_proba'):
                    final_estimator = step
                    break
        if final_estimator is None and hasattr(model, 'predict_proba'):
            # fallback: pipeline exposes predict_proba (rare)
            final_estimator = model

        if final_estimator is not None:
            explainer = shap.TreeExplainer(final_estimator)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values[1], X_train, show=False)
            plt.savefig('static/images/shap_summary.png')
            plt.close()
            print('Saved SHAP summary at static/images/shap_summary.png')
        else:
            print('No suitable estimator for SHAP found; skipping SHAP plot')
    except Exception as e:
        print('SHAP plotting failed:', e)
else:
    print('SHAP not used — shap_summary.png not produced')

# ==========================================================
# Step 8: Quick Test Prediction (unchanged)
# ==========================================================
sample = X_test.iloc[:3]
print("\n🔍 Sample predictions:")
print(model.predict(sample))
print("\n✅ Training completed successfully!")
