# ====================================
# LIAR Dataset Analysis with BERT
# ====================================

# ====================================
# 1. Setup & Installation
# ====================================
# !pip install transformers torch scikit-learn pandas matplotlib seaborn

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

train_path = "train-train-clean.csv"
valid_path = "valid-valid-clean.csv"

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
# 5. Binary Label Simplification & Encoding
# ====================================
def simplify_to_binary(label):
    """
    Simplify multi-class labels to binary:
    - 'true', 'mostly-true' → 'TRUE'
    - 'half-true', 'barely-true', 'false', 'pants-fire' → 'FALSE'
    """
    if label.lower() in ['true', 'mostly-true']:
        return 'TRUE'
    else:
        return 'FALSE'

# Apply binary simplification
train_df['label_binary'] = train_df['label'].apply(simplify_to_binary)
valid_df['label_binary'] = valid_df['label'].apply(simplify_to_binary)

print("Original label distribution:")
print(train_df['label'].value_counts())
print(f"\nBinary label distribution:")
print(train_df['label_binary'].value_counts())

# Encode binary labels
le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label_binary"])
valid_df["label_enc"] = le.transform(valid_df["label_binary"])

print(f"\nBinary classes: {list(le.classes_)}")
print(f"Class balance: {train_df['label_binary'].value_counts(normalize=True).round(3).to_dict()}")

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
# 7. Build Baseline Model: Binary Logistic Regression with BERT
# ====================================
print("Training baseline binary classification model with BERT features...")
baseline_model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",  # Good for binary classification
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
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
# 10. Train Combined Model (BERT + Metadata) - Binary
# ====================================
print("Training combined binary classification model (BERT + Metadata)...")
combined_model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",  # Good for binary classification
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
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
# 14. Feature Interpretation - Binary Classification
# ====================================
def get_top_metadata_features_binary(model, metadata_columns, top_k=10):
    """Extract top metadata features for binary classification"""
    # For binary classification, we only have one coefficient vector
    # Metadata features start after BERT embeddings (768 features for DistilBERT)
    bert_embedding_dim = 768
    metadata_coefs = model.coef_[0][bert_embedding_dim:]  # Shape: (n_metadata_features,)

    # Get top positive and negative coefficients
    top_pos_idx = np.argsort(metadata_coefs)[-top_k:][::-1]
    top_neg_idx = np.argsort(metadata_coefs)[:top_k]

    top_pos_features = [(metadata_columns[j], metadata_coefs[j]) for j in top_pos_idx if metadata_coefs[j] > 0.01]
    top_neg_features = [(metadata_columns[j], metadata_coefs[j]) for j in top_neg_idx if metadata_coefs[j] < -0.01]

    return top_pos_features, top_neg_features

# Analyze metadata feature importance for binary classification
if combined_accuracy > baseline_accuracy:
    print("\n" + "="*60)
    print("TOP METADATA FEATURES FOR BINARY CLASSIFICATION")
    print("="*60)

    pos_features, neg_features = get_top_metadata_features_binary(
        combined_model,
        meta_df_train.columns.tolist(),
        top_k=8
    )

    print(f"Features that predict TRUE (factual statements):")
    for feat, coef in pos_features:
        print(f"  {feat}: {coef:.4f}")

    print(f"\nFeatures that predict FALSE (misleading statements):")
    for feat, coef in neg_features:
        print(f"  {feat}: {coef:.4f}")

    # Additional analysis - BERT vs Metadata contribution
    print(f"\nBERT vs Metadata contribution:")
    bert_embedding_dim = 768  # DistilBERT embedding dimension
    bert_coefs = combined_model.coef_[0][:bert_embedding_dim]
    meta_coefs = combined_model.coef_[0][bert_embedding_dim:]

    print(f"  BERT features mean absolute weight: {np.mean(np.abs(bert_coefs)):.4f}")
    print(f"  Metadata features mean absolute weight: {np.mean(np.abs(meta_coefs)):.4f}")

    # Feature importance ratio
    bert_importance = np.sum(np.abs(bert_coefs))
    meta_importance = np.sum(np.abs(meta_coefs))
    total_importance = bert_importance + meta_importance

    print(f"  BERT contribution: {bert_importance/total_importance:.1%}")
    print(f"  Metadata contribution: {meta_importance/total_importance:.1%}")

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
# 16. Performance Analysis - Binary Classification
# ====================================
print("\n" + "="*60)
print("DETAILED BINARY CLASSIFICATION ANALYSIS")
print("="*60)

# Binary classification metrics
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

# Get prediction probabilities for AUC
y_proba_baseline = baseline_model.predict_proba(X_valid_bert)[:, 1]
y_proba_combined = combined_model.predict_proba(X_valid_combined)[:, 1]

# Calculate AUC scores
auc_baseline = roc_auc_score(y_valid, y_proba_baseline)
auc_combined = roc_auc_score(y_valid, y_proba_combined)

print(f"ROC-AUC Scores:")
print(f"  Baseline (BERT only):     {auc_baseline:.4f}")
print(f"  Combined (BERT + Meta):   {auc_combined:.4f}")

# Precision, Recall, F1 for each class
precision, recall, f1, support = precision_recall_fscore_support(y_valid, y_pred_combined, average=None)

print(f"\nDetailed Performance (Combined Model):")
for i, class_name in enumerate(le.classes_):
    print(f"  {class_name:5}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f} (n={support[i]})")

# Overall metrics
print(f"\nOverall Performance:")
print(f"  Macro F1:        {np.mean(f1):.4f}")
print(f"  Weighted F1:     {f1.mean():.4f}")

# ROC Curve visualization
plt.figure(figsize=(12, 5))

# ROC curves
plt.subplot(1, 2, 1)
fpr_baseline, tpr_baseline, _ = roc_curve(y_valid, y_proba_baseline)
fpr_combined, tpr_combined, _ = roc_curve(y_valid, y_proba_combined)

plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC={auc_baseline:.3f})', linewidth=2)
plt.plot(fpr_combined, tpr_combined, label=f'Combined (AUC={auc_combined:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Probability distribution
plt.subplot(1, 2, 2)
plt.hist(y_proba_combined[y_valid == 0], bins=30, alpha=0.7, label='FALSE', color='red', density=True)
plt.hist(y_proba_combined[y_valid == 1], bins=30, alpha=0.7, label='TRUE', color='blue', density=True)
plt.xlabel('Predicted Probability of TRUE')
plt.ylabel('Density')
plt.title('Prediction Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

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

print(f"\n🎯 Binary classification results:")
print(f"   Baseline accuracy: {baseline_accuracy:.3f}")
print(f"   Combined accuracy: {combined_accuracy:.3f}")

if combined_accuracy >= 0.75:
    print(f"✅ TARGET ACHIEVED! Accuracy ({combined_accuracy:.3f}) ≥ 75%")
    print("   Binary simplification was successful!")
else:
    print(f"⚠️  Close to target. Current: {combined_accuracy:.3f}, Target: 0.75")
    print(f"   Need {0.75 - combined_accuracy:.3f} more accuracy points")
    print("   Consider: fine-tuning BERT, ensemble methods, or more features")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)