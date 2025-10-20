# train_and_submit.py
import time

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Parameters (can be adjusted) ----------
TRAIN_CSV = "train_subset.csv"
TEST_CSV = "test_kaggle_features.csv"
ID_COL = "id"
TARGET = "satisfaction"
RANDOM_STATE = 42
CV_FOLDS = 5
# --------------------------------------------------

def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    return train, test

# Separate features into numeric and categorical types (excluding ID and target columns)
def infer_feature_types(df, id_col, target_col=None):
    exclude = {id_col}
    if target_col:
        exclude.add(target_col)
    cols = [c for c in df.columns if c not in exclude]
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, categorical_cols

# Build a preprocessing pipeline
def build_preprocessor(numeric_cols, categorical_cols):
    # Numeric columns: fill missing values with median, then standardize
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical columns: fill missing values with 'missing', then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine numeric and categorical pipelines into one column transformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    return preprocessor

def baseline_models():
    # Return a dictionary of baseline models
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNeighbors": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        # "SVC": SVC(probability=True, random_state=RANDOM_STATE),  # SVC is disabled (too slow)
        "GaussianNB": GaussianNB()
    }

def encode_target(y_series):
    # Encode string labels to integer values
    labels = y_series.astype(str).unique().tolist()
    labels_sorted = sorted(labels)
    label_to_int = {lab: i for i, lab in enumerate(labels_sorted)}
    y_enc = y_series.astype(str).map(label_to_int)
    return y_enc.values, label_to_int

def inverse_label_map(int_preds, label_map):
    inv = {v: k for k, v in label_map.items()}
    return [inv[int(p)] for p in int_preds]

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in training data.")

    # Infer numeric and categorical columns
    numeric_cols, categorical_cols = infer_feature_types(train, ID_COL, TARGET)
    print("\nNumeric columns:", numeric_cols)
    print("\nCategorical columns:", categorical_cols)

    # Encode target labels
    y_raw = train[TARGET]
    train_target, label_map = encode_target(y_raw)
    print("\nLabel mapping:", label_map)

    # Prepare feature matrix (drop ID and target columns)
    train_drop = train.drop(columns=[ID_COL, TARGET])
    if ID_COL in test.columns:
        x_test = test.drop(columns=[ID_COL])
    else:
        x_test = test.copy()

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Split into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        train_drop,
        train_target,
        test_size=0.2,
        stratify=train_target,
        random_state=RANDOM_STATE
    )
    print(f"\nTrain/Val shapes: {x_train.shape}/{x_val.shape}")

    # Define baseline models
    models = baseline_models()

    # Evaluate baseline models using cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_scores = {}

    for name, clf in models.items():
        print("=" * 60)
        print(f"🔍 Cross-validating {name} ...")

        pipe = Pipeline(steps=[
            ("preproc", preprocessor),
            ("clf", clf)
        ])

        # 1️⃣ Cross-validation for F1 and Accuracy
        f1_scores = cross_val_score(pipe, train_drop, train_target, cv=cv, scoring="f1_macro", n_jobs=-1)
        acc_scores = cross_val_score(pipe, train_drop, train_target, cv=cv, scoring="accuracy", n_jobs=-1)

        f1_mean, f1_std = f1_scores.mean(), f1_scores.std()
        acc_mean, acc_std = acc_scores.mean(), acc_scores.std()

        model_scores[name] = {
            "F1_mean": f1_mean,
            "F1_std": f1_std,
            "ACC_mean": acc_mean,
            "ACC_std": acc_std
        }

        print(f"{name}:")
        print(f"  ✅ mean F1_macro = {f1_mean:.4f} (+/- {f1_std:.4f})")
        print(f"  ✅ mean Accuracy = {acc_mean:.4f} (+/- {acc_std:.4f})")

        # 2️⃣ Train on training set and show confusion matrix
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_val)

        print("\n📊 Confusion matrix:")
        cm = confusion_matrix(y_val, y_pred)
        print(cm)

        print("\n📋 Classification report:")
        print(classification_report(y_val, y_pred, digits=4))

    # 3️⃣ Summary of CV scores
    print("\n\n🏁 Model performance summary (cross-validation):")
    for name, score in model_scores.items():
        print(f"{name:20s} | F1={score['F1_mean']:.4f} | ACC={score['ACC_mean']:.4f}")

    # Choose best model by F1_macro
    best_model_name = max(model_scores.items(), key=lambda kv: kv[1]['F1_mean'])[0]
    print("Best model by CV F1_macro:", best_model_name)

    # Hyperparameter tuning for RandomForest using GridSearchCV
    if "RandomForest" in models:
        print("Running GridSearchCV on RandomForest ...")
        rf_pipe = Pipeline(steps=[
            ("preproc", preprocessor),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
        ])
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5]
        }
        gs = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
        gs.fit(x_train, y_train)
        print("GridSearch best params:", gs.best_params_)
        print("GridSearch best score (F1_macro):", gs.best_score_)

        if gs.best_score_ > model_scores[best_model_name]['F1_mean']:
            print("GridSearch RandomForest outperforms previous best. Using GridSearch RF as final model.")
            final_pipeline = gs.best_estimator_
            final_name = "RandomForest_GridSearch"
        else:
            final_pipeline = Pipeline(steps=[
                ("preproc", preprocessor),
                ("clf", models[best_model_name])
            ])
            final_pipeline.fit(x_train, y_train)
            final_name = best_model_name
    else:
        final_pipeline = Pipeline(steps=[
            ("preproc", preprocessor),
            ("clf", models[best_model_name])
        ])
        final_pipeline.fit(x_train, y_train)
        final_name = best_model_name

    print("Final model selected:", final_name)

    # Evaluate on held-out validation set
    print("Evaluating on held-out validation set...")
    y_val_pred = final_pipeline.predict(x_val)
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    print(f"Validation Accuracy: {acc:.4f}, F1_macro: {f1:.4f}")
    print("Classification report (validation):")

    y_val_pred_labels = inverse_label_map(y_val_pred, label_map)
    y_val_labels = inverse_label_map(y_val, label_map)
    print(classification_report(y_val_labels, y_val_pred_labels))

    # Predict on test set
    print("Predicting on test set...")
    test_pred_int = final_pipeline.predict(x_test)
    test_pred_labels = inverse_label_map(test_pred_int, label_map)

    # Create submission file
    submission = pd.DataFrame({
        "ID": test[ID_COL] if ID_COL in test.columns else np.arange(len(test_pred_labels)),
        "label": test_pred_labels
    })
    submission.to_csv("my_submission.csv", index=False)
    print("Saved my_submission.csv")

    # Save final model
    joblib.dump(final_pipeline, "final_model_pipeline.joblib")
    print("Saved final_model_pipeline.joblib")

if __name__ == "__main__":
    print("run time:", time.time())
    main()
