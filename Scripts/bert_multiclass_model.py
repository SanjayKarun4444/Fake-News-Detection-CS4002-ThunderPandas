"""
===============================================================================
bert_multiclass_model.py

Purpose:
    Fine-tunes a BERT-based model for multiclass fake news classification.
    Predicts the statement label as one of six LIAR truthfulness levels.

Authors:
    Team Thunder Pandas (DS4002, Fall 2025)

Usage:
    python bert_multiclass_model.py

Inputs:
    - Preprocessed LIAR dataset CSV/TSV files
    - Model hyperparameters

Outputs:
    - Trained multiclass BERT model
    - Evaluation metrics (accuracy, recall, F1)
    - Confusion matrix plots to OUTPUT/

Requirements:
    - Python 3.8+
    - torch, transformers, pandas, numpy, scikit-learn, matplotlib, tqdm

Notes:
    - Script includes extensive comments for clarity.
    - See docstrings and block comments for each major section.

===============================================================================
"""


# ====================================
# LIAR Dataset Analysis with BERT
# ====================================

# ====================================
# 1. Setup & Installation
# ====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import binomtest
from scipy.sparse import csr_matrix, hstack
import pickle
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ====================================
# 2. Load Data
# ====================================
# Upload your files to Colab first using:
# from google.colab import files
# uploaded = files.upload()

train_path = "Data/train-train-clean.csv"
valid_path = "Data/valid-valid-clean.csv"

# Load CSV files
train_df = pd.read_csv(train_path, header=None, engine="python")
valid_df = pd.read_csv(valid_path, header=None, engine="python")

print("Train shape:", train_df.shape)
print("Valid shape:", valid_df.shape)

# ====================================
# 3. Assign Column Names
# ====================================
cols = [
    "id", "label", "statement",
    "speaker", "title", "state", "party",
    "num1", "num2", "num3", "num4", "num5",
    "context"
]
train_df.columns = cols
valid_df.columns = cols

print("Sample data:")
print(train_df[['label', 'statement', 'party', 'state']].head())

# ====================================
# 4. Clean and Process Text Data
# ====================================
def clean_text(text):
    """Clean text by handling NaN and basic preprocessing"""
    if pd.isna(text):
        return ""
    return str(text).strip()

train_df['statement_clean'] = train_df['statement'].apply(clean_text)
valid_df['statement_clean'] = valid_df['statement'].apply(clean_text)

print(f"Sample cleaned statements:")
for i in range(3):
    print(f"{i+1}: {train_df['statement_clean'].iloc[i][:100]}...")

# ====================================
# 5. Encode Labels
# ====================================
le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label"])
valid_df["label_enc"] = le.transform(valid_df["label"])

print("Label classes:", list(le.classes_))
print("Label distribution:")
print(train_df['label'].value_counts())

# ====================================
# 6. BERT Text Encoding
# ====================================
# Use DistilBERT (smaller, faster than full BERT)
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)
bert_model.to(device)
bert_model.eval()

def get_bert_embeddings(texts, batch_size=16, max_length=512):
    """
    Get BERT embeddings for a list of texts
    Returns: numpy array of shape (len(texts), 768) for DistilBERT
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move to device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Use [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

print("Generating BERT embeddings for training data...")
X_train_bert = get_bert_embeddings(train_df['statement_clean'].tolist())
print(f"Training embeddings shape: {X_train_bert.shape}")

print("Generating BERT embeddings for validation data...")
X_valid_bert = get_bert_embeddings(valid_df['statement_clean'].tolist())
print(f"Validation embeddings shape: {X_valid_bert.shape}")

y_train = train_df["label_enc"].values
y_valid = valid_df["label_enc"].values

# ====================================
# 7. Build Baseline Model: Logistic Regression with BERT
# ====================================
print("Training baseline model with BERT features...")
baseline_model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",  # Good for smaller datasets
    multi_class="multinomial",
    random_state=42
)

baseline_model.fit(X_train_bert, y_train)
y_pred_baseline = baseline_model.predict(X_valid_bert)

baseline_accuracy = accuracy_score(y_valid, y_pred_baseline)
print(f"\nBaseline BERT Model Results:")
print(f"Accuracy: {baseline_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_baseline, target_names=le.classes_))

# ====================================
# 8. Confusion Matrix for Baseline
# ====================================
plt.figure(figsize=(12, 8))
cm_baseline = confusion_matrix(y_valid, y_pred_baseline)
sns.heatmap(cm_baseline, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Baseline BERT Model")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ====================================
# 9. Supplement with Metadata Features
# ====================================
print("Adding metadata features...")

# Create metadata features (one-hot encoding)
meta_df_train = pd.get_dummies(
    train_df[["party", "state", "context"]].fillna("unknown"),
    drop_first=True
).astype(float)

meta_df_valid = pd.get_dummies(
    valid_df[["party", "state", "context"]].fillna("unknown"),
    drop_first=True
).astype(float)

# Align columns between train and validation
meta_df_valid = meta_df_valid.reindex(columns=meta_df_train.columns, fill_value=0.0)

print(f"Metadata features shape - Train: {meta_df_train.shape}, Valid: {meta_df_valid.shape}")
print(f"Sample metadata columns: {meta_df_train.columns[:10].tolist()}")

# Combine BERT embeddings with metadata
X_train_combined = np.hstack([X_train_bert, meta_df_train.values])
X_valid_combined = np.hstack([X_valid_bert, meta_df_valid.values])

print(f"Combined features shape - Train: {X_train_combined.shape}, Valid: {X_valid_combined.shape}")

# ====================================
# 10. Train Combined Model (BERT + Metadata)
# ====================================
print("Training combined model (BERT + Metadata)...")
combined_model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    multi_class="multinomial",
    random_state=42
)

combined_model.fit(X_train_combined, y_train)
y_pred_combined = combined_model.predict(X_valid_combined)

combined_accuracy = accuracy_score(y_valid, y_pred_combined)
print(f"\nCombined Model Results:")
print(f"Accuracy: {combined_accuracy:.4f}")
print(f"Improvement over baseline: {combined_accuracy - baseline_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_combined, target_names=le.classes_))

# ====================================
# 11. Confusion Matrix for Combined Model
# ====================================
plt.figure(figsize=(12, 8))
cm_combined = confusion_matrix(y_valid, y_pred_combined)
sns.heatmap(cm_combined, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Combined BERT + Metadata Model")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ====================================
# 12. Model Comparison
# ====================================
print("="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"Baseline (BERT only):           {baseline_accuracy:.4f}")
print(f"Combined (BERT + Metadata):     {combined_accuracy:.4f}")
print(f"Metadata improvement:           {combined_accuracy - baseline_accuracy:.4f}")
print("="*50)

# ====================================
# 13. Hypothesis Testing
# ====================================
n_samples = len(y_valid)
correct_baseline = sum(y_pred_baseline == y_valid)
correct_combined = sum(y_pred_combined == y_valid)

# H0: accuracy ≤ 0.75, H1: accuracy > 0.75
test_baseline = binomtest(correct_baseline, n_samples, p=0.75, alternative="greater")
test_combined = binomtest(correct_combined, n_samples, p=0.75, alternative="greater")

print("\nHypothesis Testing (H0: accuracy ≤ 75%, H1: accuracy > 75%):")
print(f"Baseline model: {correct_baseline}/{n_samples} correct, p-value = {test_baseline.pvalue:.6f}")
print(f"Combined model: {correct_combined}/{n_samples} correct, p-value = {test_combined.pvalue:.6f}")

if test_combined.pvalue < 0.05:
    print("✅ Reject H0: Model accuracy significantly > 75%")
else:
    print("❌ Fail to reject H0: No evidence that accuracy > 75%")

# ====================================
# 14. Feature Interpretation
# ====================================
def get_top_metadata_features(model, metadata_columns, n_classes, top_k=10):
    """Extract top metadata features for each class"""
    # Metadata features start after BERT embeddings (768 features)
    bert_dim = 768
    metadata_coefs = model.coef_[:, bert_dim:]  # Shape: (n_classes, n_metadata_features)

    results = {}
    for i, class_name in enumerate(le.classes_):
        # Get coefficients for this class
        class_coefs = metadata_coefs[i]

        # Get top positive and negative coefficients
        top_pos_idx = np.argsort(class_coefs)[-top_k:][::-1]
        top_neg_idx = np.argsort(class_coefs)[:top_k]

        top_pos_features = [(metadata_columns[j], class_coefs[j]) for j in top_pos_idx]
        top_neg_features = [(metadata_columns[j], class_coefs[j]) for j in top_neg_idx]

        results[class_name] = {
            'positive': top_pos_features,
            'negative': top_neg_features
        }

    return results

# Analyze metadata feature importance
if combined_accuracy > baseline_accuracy:
    print("\n" + "="*60)
    print("TOP METADATA FEATURES BY CLASS")
    print("="*60)

    feature_importance = get_top_metadata_features(
        combined_model,
        meta_df_train.columns.tolist(),
        len(le.classes_),
        top_k=5
    )

    for class_name, features in feature_importance.items():
        print(f"\nClass: {class_name}")
        print("  Most Predictive (Positive):")
        for feat, coef in features['positive']:
            print(f"    {feat}: {coef:.4f}")
        print("  Most Predictive (Negative):")
        for feat, coef in features['negative'][:3]:  # Show fewer negative
            print(f"    {feat}: {coef:.4f}")

# ====================================
# 15. Save Models and Results
# ====================================
artifacts = {
    'tokenizer': tokenizer,
    'bert_model_name': model_name,
    'label_encoder': le,
    'baseline_model': baseline_model,
    'combined_model': combined_model,
    'metadata_columns': meta_df_train.columns.tolist(),
    'baseline_accuracy': baseline_accuracy,
    'combined_accuracy': combined_accuracy,
    'results': {
        'baseline_predictions': y_pred_baseline,
        'combined_predictions': y_pred_combined,
        'true_labels': y_valid
    }
}

with open('liar_bert_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\n✅ Models and results saved to 'liar_bert_artifacts.pkl'")

# ====================================
# 16. Performance Analysis by Class
# ====================================
print("\n" + "="*60)
print("DETAILED PERFORMANCE ANALYSIS")
print("="*60)

# Per-class accuracy
from sklearn.metrics import classification_report
report_dict = classification_report(y_valid, y_pred_combined, target_names=le.classes_, output_dict=True)

print("Per-class Performance (Combined Model):")
for class_name in le.classes_:
    if class_name in report_dict:
        precision = report_dict[class_name]['precision']
        recall = report_dict[class_name]['recall']
        f1 = report_dict[class_name]['f1-score']
        support = report_dict[class_name]['support']
        print(f"  {class_name:12}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (n={int(support)})")

# ====================================
# 17. Next Steps and Recommendations
# ====================================
print("\n" + "="*60)
print("NEXT STEPS FOR IMPROVEMENT")
print("="*60)
print("1. Fine-tune BERT: Instead of using frozen BERT + LogReg, fine-tune BERT end-to-end")
print("2. Try different models: RoBERTa, ELECTRA, or domain-specific models")
print("3. Experiment with different pooling strategies (mean, max, attention)")
print("4. Add more metadata features: speaker history, statement length, etc.")
print("5. Handle class imbalance: use class weights or SMOTE")
print("6. Ensemble methods: combine multiple models")

if combined_accuracy < 0.75:
    print(f"\n⚠️  Current accuracy ({combined_accuracy:.3f}) is below target (0.75)")
    print("   Consider label consolidation (e.g., binary true/false) for higher accuracy")
else:
    print(f"\n🎯 Target accuracy achieved! ({combined_accuracy:.3f} > 0.75)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)