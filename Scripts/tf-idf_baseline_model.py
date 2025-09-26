"""
===============================================================================
tf-idf_baseline_model.py

Purpose:
    Trains and evaluates a logistic regression model using TF-IDF features
    for fake news classification (baseline). Provides a reference for model
    performance using classic ML, compared with BERT models.

Authors:
    Team Thunder Pandas (DS4002, Fall 2025)

Usage:
    python tf-idf_baseline_model.py

Inputs:
    - Cleaned LIAR dataset CSV/TSV files

Outputs:
    - Trained scikit-learn model (optional/manual save)
    - Baseline performance metrics
    - Confusion matrix visualizations to OUTPUT/

Requirements:
    - Python 3.8+
    - scikit-learn, pandas, numpy, matplotlib

Notes:
    - Copious inline comments throughout enhance reproducibility.
    - Block comments mark each functional section.

===============================================================================
"""



# ====================================
# 1. Setup
# ====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from scipy.stats import binomtest
import pickle
# ====================================
# 2. Load Data
# ====================================
# NOTE: Replace these with your own Colab upload paths
# You can upload by running:
# from google.colab import files
# uploaded = files.upload()

train_path = "Data/train-train-clean.csv"   # update if your filename differs
valid_path = "Data/valid-valid-clean.csv"

# Load CSV (no headers in LIAR dataset)
train_df = pd.read_csv(train_path, header=None, engine="python")
valid_df = pd.read_csv(valid_path, header=None, engine="python")

print("Train shape:", train_df.shape)
print("Valid shape:", valid_df.shape)
train_df.head()


# ====================================
# 3. Assign column names (heuristic)
# ====================================
cols = [
    "id", "label", "statement",
    "speaker", "title", "state", "party",
    "num1", "num2", "num3", "num4", "num5",
    "context"
]
train_df.columns = cols
valid_df.columns = cols

print(train_df.sample(3))

# ====================================
# 4. Encode labels
# ====================================
le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label"])
valid_df["label_enc"] = le.transform(valid_df["label"])

print("Classes:", list(le.classes_))


# ====================================
# 5. TF-IDF Baseline Model
# ====================================
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000
)

X_train = vectorizer.fit_transform(train_df["statement"].fillna(""))
X_valid = vectorizer.transform(valid_df["statement"].fillna(""))

y_train = train_df["label_enc"].values
y_valid = valid_df["label_enc"].values

baseline_model = LogisticRegression(
    max_iter=2000,
    solver="saga",
    multi_class="multinomial"
)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_valid)

print("Baseline TF-IDF only")
print("Accuracy:", accuracy_score(y_valid, y_pred))
print(classification_report(y_valid, y_pred, target_names=le.classes_))


# ====================================
# 6. Confusion Matrix
# ====================================
cm = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Baseline TF-IDF")
plt.show()



# ====================================
# 7. Add Metadata Features (fixed)
# ====================================
# One-hot encode categorical metadata (party, state, context)
meta_df_train = pd.get_dummies(
    train_df[["party", "state", "context"]].fillna("unknown"),
    drop_first=True
).astype(float)

meta_df_valid = pd.get_dummies(
    valid_df[["party", "state", "context"]].fillna("unknown"),
    drop_first=True
).astype(float)

# Align columns (important!)
meta_df_valid = meta_df_valid.reindex(columns=meta_df_train.columns, fill_value=0)

# Combine TF-IDF + metadata
from scipy.sparse import hstack
X_train_combined = hstack([X_train, meta_df_train.values])
X_valid_combined = hstack([X_valid, meta_df_valid.values])

combined_model = LogisticRegression(
    max_iter=2000,
    solver="saga",
    multi_class="multinomial"
)
combined_model.fit(X_train_combined, y_train)

y_pred_meta = combined_model.predict(X_valid_combined)

print("TF-IDF + Metadata")
print("Accuracy:", accuracy_score(y_valid, y_pred_meta))
print(classification_report(y_valid, y_pred_meta, target_names=le.classes_))



# ====================================
# 8. Hypothesis Test
# H0: Accuracy <= 75%
# H1: Accuracy > 75%
# ====================================
n = len(y_valid)
k_baseline = sum(y_pred == y_valid)
k_meta = sum(y_pred_meta == y_valid)

test_baseline = binomtest(k_baseline, n, p=0.75, alternative="greater")
test_meta = binomtest(k_meta, n, p=0.75, alternative="greater")

print("Baseline accuracy:", k_baseline / n, "p-value:", test_baseline.pvalue)
print("Meta accuracy:", k_meta / n, "p-value:", test_meta.pvalue)



# ====================================
# 9. Save Models/Artifacts (optional)
# ====================================
artifacts = {
    "vectorizer": vectorizer,
    "label_encoder": le,
    "baseline_model": baseline_model,
    "combined_model": combined_model,
    "meta_columns": meta_df_train.columns.tolist()
}

# with open("liar_artifacts.pkl", "wb") as f:
#     pickle.dump(artifacts, f)

# print("Artifacts saved as liar_artifacts.pkl")



# Top TF-IDF words per class (baseline)
feature_names = vectorizer.get_feature_names_out()
coefs = baseline_model.coef_  # shape (n_classes, n_features)
topk = 20
for i, cls in enumerate(le.classes_):
    top_idx = np.argsort(coefs[i])[-topk:][::-1]
    top_words = [feature_names[j] for j in top_idx]
    print(f"\nClass: {cls} — top {topk} words:")
    print(top_words[:30])

# Top features from combined model (TF-IDF + metadata)
if 'combined_model' in globals():
    tfidf_count = len(feature_names)
    meta_names = list(meta_df_train.columns)
    combined_names = list(feature_names) + meta_names
    coefs_comb = combined_model.coef_
    for i, cls in enumerate(le.classes_):
        top_idx = np.argsort(coefs_comb[i])[-30:][::-1]
        top_feats = [(combined_names[j], float(coefs_comb[i, j])) for j in top_idx]
        # split into textual and metadata for readability
        top_text = [name for name, coef in top_feats if combined_names.index(name) < tfidf_count][:15]
        top_meta = [name for name, coef in top_feats if combined_names.index(name) >= tfidf_count][:15]
        print(f"\nClass {cls}: top textual words (sample): {top_text[:10]}")
        print(f"Class {cls}: top metadata features (sample): {top_meta[:10]}")

