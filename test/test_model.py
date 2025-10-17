#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_model.py

功能：
1. 加载训练好的 Pipeline 模型 (joblib)
2. 读取新数据 CSV
3. 输出预测类别和概率
4. 将预测结果保存到 CSV
"""

import joblib
import pandas as pd

# -------------------------------
# 配置
# -------------------------------
MODEL_PATH = "../final_model_pipeline.joblib"  # 保存的模型文件
INPUT_CSV = "test_model_data.csv"                  # 新数据 CSV 文件
OUTPUT_CSV = "test_model_data_result.csv"              # 保存预测结果 CSV
LABEL_MAPPING = {0: "neutral or dissatisfied", 1: "satisfied"}  # 标签映射

# -------------------------------
# 1️⃣ 加载模型
# -------------------------------
print(f"Loading model from {MODEL_PATH} ...")
model_pipeline = joblib.load(MODEL_PATH)
print("Model loaded successfully.\n")

# -------------------------------
# 2️⃣ 读取新数据
# -------------------------------
print(f"Reading new data from {INPUT_CSV} ...")
try:
    new_data = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: File {INPUT_CSV} not found!")
    exit(1)

print(f"New data shape: {new_data.shape}\n")

# -------------------------------
# 3️⃣ 预测类别和概率
# -------------------------------
print("Predicting classes and probabilities ...")
y_pred = model_pipeline.predict(new_data)
y_prob = model_pipeline.predict_proba(new_data)  # 返回 n_samples x n_classes

# 映射类别
y_pred_labels = [LABEL_MAPPING[c] for c in y_pred]

# -------------------------------
# 4️⃣ 输出预测结果
# -------------------------------
results_df = new_data.copy()
results_df["pred_class"] = y_pred_labels
results_df["prob_0"] = y_prob[:, 0]  # 对应 0 类
results_df["prob_1"] = y_prob[:, 1]  # 对应 1 类

print("\nPrediction example:")
print(results_df.head())

# -------------------------------
# 5️⃣ 保存到 CSV
# -------------------------------
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nPredictions saved to {OUTPUT_CSV}")
