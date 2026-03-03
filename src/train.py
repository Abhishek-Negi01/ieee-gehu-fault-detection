# Trains Stacking Ensemble (LightGBM + XGBoost + ExtraTrees)


# import required things
import numpy as np
import pandas as pd
import pickle
import os

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, log_loss, average_precision_score
)

# add directories
DATA_PATH  = "TRAIN.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "stacking_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# prepare labels
X = df.drop('Class',axis=1)
y = df['Class']

# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# scale pos weight
neg, pos         = np.bincount(y_train)
scale_pos_weight = neg if pos == 0 else neg / pos

# define base model
lgb_model = lgb.LGBMClassifier(
    n_estimators = 500,
    num_leaves = 63,
    max_depth = -1,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = scale_pos_weight,
    device = 'cpu',
    n_jobs = -1,
    random_stat = 42,
    verbose = -1
)

xgb_model = xgb.XGBClassifier(
    n_estimators = 500,
    max_depth = 8,
    learning_rate = 0.05,
    subsample = 0.9,
    colsample_bytree = 0.8,
    scale_pos_weight = scale_pos_weight,
    device = 'cuda',
    eval_metric = 'logloss',
    random_state = 42,
    verbosity = 0
)

et_model = ExtraTreesClassifier(
    n_estimators = 200,
    max_features = 'sqrt',
    class_weight = 'balanced',
    bootstrap = False,
    random_state = 42,
    n_jobs = -1
)

# stacking classifier
stack_model = StackingClassifier(
    estimators = [
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('et',  et_model)
    ],
    final_estimator = LogisticRegression(
        C = 1.0,
        class_weight = 'balanced',
        max_iter = 5000,
        solver = 'lbfgs',
        random_state = 42
    ),
    cv = 5,
    stack_method = 'predict_proba',
    passthrough = True,
    n_jobs = -1,
    verbose = 1
)


# train model
print(f"\n{'-'*100}")
print("\nTraining Stacking Ensemble (LGB + XGB + ET)...")
stack_model.fit(X_train, y_train)
print("Training complete.")

# evaluate
y_pred = stack_model.predict(X_test)
y_pred_proba = stack_model.predict_proba(X_test)[:, 1]

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'-'*100}")
print(f"  Stacking Ensemble - METRICS REPORT")
print(f"{'-'*100}")

metrics_report = {
    "Accuracy" : accuracy_score(y_test, y_pred),
    "Precision" : precision_score(y_test, y_pred),
    "Recall (Sensitivity)" : recall_score(y_test, y_pred),
    "Specificity (TNR)" : tn / (tn + fp),
    "F1 Score" : f1_score(y_test, y_pred),
    "ROC-AUC" : roc_auc_score(y_test, y_pred_proba),
    "Average Precision (PR-AUC)" : average_precision_score(y_test, y_pred_proba),
    "MCC" : matthews_corrcoef(y_test, y_pred),
    "Log Loss" : log_loss(y_test, y_pred_proba),
}

for k, v in metrics_report.items():
        print(f"  {k:<35}: {v:.4f}")

print(f"\n  TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"], digits=4))

# save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(stack_model, f)

print(f"Model saved to: {MODEL_PATH}")