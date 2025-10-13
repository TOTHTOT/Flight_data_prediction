# train_and_submit.py
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
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# ---------- 参数（可调整） ----------
TRAIN_CSV = "train_subset.csv"
TEST_CSV = "test_kaggle_features.csv"
ID_COL = "id"  # 如果你的 id 列不是 'id'，请修改
TARGET = "satisfaction"
RANDOM_STATE = 42
CV_FOLDS = 5
# ------------------------------------

def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    return train, test

def infer_feature_types(df, id_col, target_col=None):
    # 返回 numeric_cols, categorical_cols（排除 id 和 target）
    exclude = {id_col}
    if target_col:
        exclude.add(target_col)
    cols = [c for c in df.columns if c not in exclude]
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols):
    # 数值列：先用中位数填充，再标准化
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # 类别列：缺失填充为字符串 'missing'，再 one-hot（drop='if_binary' optional）
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    return preprocessor

def baseline_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNeighbors": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "SVC": SVC(probability=True, random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB()
    }

def encode_target(y_series):
    # 保留原字符串标签，同时把它编码为 0..K-1 以便 sklearn 使用
    # 返回 (y_encoded, label_encoder_dict)
    labels = y_series.astype(str).unique().tolist()
    labels_sorted = sorted(labels)  # for reproducibility
    label_to_int = {lab: i for i, lab in enumerate(labels_sorted)}
    y_enc = y_series.astype(str).map(label_to_int)
    return y_enc.values, label_to_int

def inverse_label_map(int_preds, label_map):
    # label_map: {label_str: int}, invert
    inv = {v: k for k, v in label_map.items()}
    return [inv[int(p)] for p in int_preds]

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in training data.")

    # 基本列类型判断
    numeric_cols, categorical_cols = infer_feature_types(train, ID_COL, TARGET)
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    # 目标编码
    y_raw = train[TARGET]
    y, label_map = encode_target(y_raw)
    print("Label mapping:", label_map)

    # 特征矩阵
    X = train.drop(columns=[ID_COL, TARGET])
    X_test = test.drop(columns=[ID_COL]) if ID_COL in test.columns else test.copy()

    # 建预处理器
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # 拆分验证集（保持类别比例）
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train/Val shapes: {X_train.shape}/{X_val.shape}")

    # 定义候选模型
    models = baseline_models()

    # 用 cross_val_score 比较模型（使用 pipeline 将预处理拼入）
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_scores = {}
    for name, clf in models.items():
        pipe = Pipeline(steps=[("preproc", preprocessor), ("clf", clf)])
        print(f"CV scoring for {name} ...")
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        model_scores[name] = (scores.mean(), scores.std())
        print(f"{name}: mean F1_macro = {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 选择在 CV 上表现最好的模型（按 mean F1_macro）
    best_model_name = max(model_scores.items(), key=lambda kv: kv[1][0])[0]
    print("Best model by CV F1_macro:", best_model_name)

    # 对部分模型做简单的 GridSearch（以 RandomForest 为例）
    if "RandomForest" in models:
        print("Running GridSearchCV on RandomForest (example)...")
        rf_pipe = Pipeline(steps=[("preproc", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5]
        }
        gs = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
        gs.fit(X, y)
        print("GridSearch best params:", gs.best_params_)
        print("GridSearch best score (F1_macro):", gs.best_score_)
        # 如果 GridSearch 的最佳比之前最好的还好，就选它
        if gs.best_score_ > model_scores[best_model_name][0]:
            print("GridSearch RandomForest beats previous best. Selecting GridSearch RF as final model.")
            final_pipeline = gs.best_estimator_
            final_name = "RandomForest_GridSearch"
        else:
            # 否则用之前选出的
            final_pipeline = Pipeline(steps=[("preproc", preprocessor), ("clf", models[best_model_name])])
            final_pipeline.fit(X, y)
            final_name = best_model_name
    else:
        final_pipeline = Pipeline(steps=[("preproc", preprocessor), ("clf", models[best_model_name])])
        final_pipeline.fit(X, y)
        final_name = best_model_name

    # 如果 final_pipeline 还没 fit（比如选用 GridSearch 的 gs.best_estimator_ 已经 fit），确保我们有拟合模型
    try:
        # 若 estimator 已经 fit，上面一定有；否则 fit
        if not hasattr(final_pipeline, "predict"):
            final_pipeline.fit(X, y)
    except Exception:
        final_pipeline.fit(X, y)

    print("Final model selected:", final_name)

    # 在保留的验证集上评估
    print("Evaluating on held-out validation set...")
    y_val_pred = final_pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    print(f"Validation Accuracy: {acc:.4f}, F1_macro: {f1:.4f}")
    print("Classification report (validation):")
    # 把预测 int 转回原始标签字符串以便可读性
    y_val_pred_labels = inverse_label_map(y_val_pred, label_map)
    y_val_labels = inverse_label_map(y_val, label_map)
    print(classification_report(y_val_labels, y_val_pred_labels))

    # 对测试集做预测
    print("Predicting on test set...")
    test_pred_int = final_pipeline.predict(X_test)
    test_pred_labels = inverse_label_map(test_pred_int, label_map)

    # 生成 submission
    submission = pd.DataFrame({
        "ID": test[ID_COL] if ID_COL in test.columns else np.arange(len(test_pred_labels)),
        "label": test_pred_labels
    })
    submission.to_csv("my_submission.csv", index=False)
    print("Saved my_submission.csv")

    # 保存模型以备答辩/复现
    joblib.dump(final_pipeline, "final_model_pipeline.joblib")
    print("Saved final_model_pipeline.joblib")

if __name__ == "__main__":
    main()

