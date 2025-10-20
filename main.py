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

# 排除 id(在项目中没有实际意义)和target(这是结果 )后对数据进行分类 numeric_cols 数字类型, categorical_cols 非数字类型, 字符串之类的
def infer_feature_types(df, id_col, target_col=None):
    exclude = {id_col}
    if target_col:
        exclude.add(target_col)
    # 遍历整个表格, 把排除id和target的列都取出来, 这两个值不需要参与训练, id号无意义, target是结果
    cols = [c for c in df.columns if c not in exclude]
    # 取出列表中所有属性为数字的列, 比如 123, 12.3 这样的, 通过 np.number 限定
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    # print("数字\n", numeric_cols)
    # 取出不是数字类型的列
    categorical_cols = [c for c in cols if c not in numeric_cols]
    # print("非数字\n", categorical_cols)
    return numeric_cols, categorical_cols

# 使用 Pipeline 封装工作流水线
def build_preprocessor(numeric_cols, categorical_cols):
    # 数值列：先用中位数填充，再标准化, 这里是个"流水线"按照你给定的步骤执行, 这里会封装两个步骤
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # 填充缺失值, 正常老师给的数据都没问题, 这一行基本没用,这个保证如果数据无效使用中间值填充
        ("scaler", StandardScaler())
    ])
    # 类别列：缺失填充为字符串 'missing'，再 one-hot（drop='if_binary' optional）
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")), # 和上面差不多
        # 将字符串转为数字映射表, handle_unknown 用于控制当预测集中出现训练集中不存在的数据就忽略, sparse_output 控制是否用洗漱矩阵输出数据, 这里false表示返回numpy数组方便查看
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 将步骤组合转为纯数字矩阵方便训练, 这里将 Pipeline 封装的流水线与数字列和分类列和对应, 后面训练时使用
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    return preprocessor

def baseline_models():
    # 返回一个字典, 其中定义了模型的基本参数, 老师要求的算法在scikit-learn中对应的方法, 具体模型原理和实现方法可以问ai
    # | 模型 | 类别 | 特点 | 优点 | 缺点 |
    # | -------------------------------- | ----- | --------- | ---------- | ---------- |
    # | ** LogisticRegression ** | 线性模型 | 线性决策边界 | 简单快速，可解释性强 | 处理复杂关系能力弱 |
    # | ** KNeighborsClassifier(KNN) ** | 基于距离 | 看“邻居”多数类别 | 无需训练，概念直观 | 大数据慢，对噪声敏感 |
    # | ** DecisionTreeClassifier ** | 树模型 | 规则分割特征空间 | 可解释，非线性 | 容易过拟合 |
    # | ** RandomForestClassifier ** | 集成树模型 | 多棵树投票 | 泛化能力强，鲁棒性好 | 训练较慢 |
    # | ** SVC(Support Vector Machine) ** | 核方法 | 找最优分界面 | 对高维数据强大 | 参数敏感，慢 |
    # | ** GaussianNB(朴素贝叶斯) ** | 概率模型 | 独立特征假设 | 快速，适合文本分类 | 精度较低，假设过强 |
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), # 要求的 逻辑回归 算法
        "KNeighbors": KNeighborsClassifier(),   # 要求的 KNN
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),  # 要求的 决策树
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), # 要求的 随机森林
        # "SVC": SVC(probability=True, random_state=RANDOM_STATE),    # 要求的 支持向量机, 不用这个, 算的很慢
        "GaussianNB": GaussianNB()  # 要求的 朴素贝叶斯
    }

def encode_target(y_series):
    # 保留原字符串标签，同时把它编码为 0..K-1 以便 sklearn 使用, 比如 "neutral or dissatisfied" "satisfied" 编码为 0 和 1 这样方便使用
    # 返回 (y_encoded, label_encoder_dict)
    labels = y_series.astype(str).unique().tolist() # 取出 satisfaction 这一列字符串中的唯一值变成列表
    labels_sorted = sorted(labels)  # 对上面的结果做排序 , 按照字母表顺序
    label_to_int = {lab: i for i, lab in enumerate(labels_sorted)} # 变成字典即: {'neutral or dissatisfied': 0, 'satisfied': 1]
    # 解开注释就可以看到对应的值, label_to_int是个字典
    # print("\nlabel_to_int:", label_to_int)
    y_enc = y_series.astype(str).map(label_to_int) # 将原始的列中字符串改为整数值, 即上面 label_to_int,为了方便计算
    # 打印前10个数据看看转化的对不对
    # print("\ny_enc: ", y_enc.head(10))
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

    # 基本列类型判断, 这里不会修改原始的训练集
    numeric_cols, categorical_cols = infer_feature_types(train, ID_COL, TARGET)
    print("\nNumeric cols:", numeric_cols)
    print("\nCategorical cols:", categorical_cols)

    # 目标编码, 按照需求就是 satisfaction 这一列数据
    y_raw = train[TARGET]
    train_target, label_map = encode_target(y_raw)
    print("\nLabel mapping:", label_map)
    # print("\nTrain_target:", train_target)
    # 特征矩阵, 这里去掉id列和satisfaction列, 实际训练的时候只有特征数据有用
    train_drop = train.drop(columns=[ID_COL, TARGET])
    # 如果测试集也有id的话也删除
    if ID_COL in test.columns:
        x_test = test.drop(columns=[ID_COL])
    else:
        x_test = test.copy()

    # 建预处理器
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # 拆分验证集, 按照 test_size 值拆分, 这里的功能用于验证老师要求的模型, 先拆分训练集然后用baseline_models()里的模型计算训练集数据从而拿到最适合的模型用于后续预测
    # | 变量名 | 含义 |
    # | --------- | --------------- |
    # | `x_train` | 用于训练模型的输入数据 |
    # | `x_val` | 用于验证（测试）模型的输入数据 |
    # | `y_train` | 训练集对应的标签 |
    # | `y_val` | 验证集对应的标签 |
    x_train, x_val, y_train, y_val = train_test_split(train_drop, # 训练集, 前面去除了 id 和 target的结果;
                                                      train_target, # 目标, 也就是 satisfaction, train_drop 和 y一一对应;
                                                      test_size=0.2, # 表示验证集占全部数据的 20%, 训练集占 80%.
                                                      stratify=train_target, # 拆分数据, 在拆分数据时, 按照标签 train_target 的分布比例来分割数据, 如果不这样可能会导致验证集比例失衡
                                                      # 设置随机值, 粗浅可以理解为 train_test_split() 在随机抽取样本时的行为是随机的,
                                                      # 这里设置固定值, 让他行为不随机保证在不同电脑都能得到相同结果
                                                      random_state=RANDOM_STATE)
    print(f"\nTrain/Val shapes: {x_train.shape}/{x_val.shape}")

    # 定义候选模型
    models = baseline_models()

    # 用候选模型分别计算训练集, 看看那个模型最好, 其实可以不用这么做 随便选一个也行, 但是不同模型适合的数据场景有优劣, 选出最好的模型可以得到最好的测试结果
    # 这里按照老师的要求测试不同算法的准确率, F1-score, 混淆矩阵
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_scores = {} # 每次计算的结果保存在里面, 后面比较分数最高地用于本次训练和验证
    for name, clf in models.items():
        print("=" * 60)
        print(f"🔍 交叉验证评估 {name} ...")

        # 构建 pipeline（预处理 + 模型）, preprocessor 是 build_preprocessor()
        # 的结果包含了表格中不同列执行的流程 categorical_transformer 和 numeric_transformer
        pipe = Pipeline(steps=[
            ("preproc", preprocessor),
            ("clf", clf)
        ])

        # ----- 1️⃣ F1-score 交叉验证 -----
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

        # ----- 2️⃣ 在训练集上训练并生成混淆矩阵 -----
        pipe.fit(x_train, y_train) # 训练
        y_pred = pipe.predict(x_val) # 预测

        print("\n📊 混淆矩阵：")
        cm = confusion_matrix(y_val, y_pred)
        print(cm)

        # 打印详细报告（Precision, Recall, F1）
        print("\n📋 分类报告：")
        print(classification_report(y_val, y_pred, digits=4))

        # 绘制混淆矩阵图, 生成视图 可以不使用
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        # plt.title(f"Confusion Matrix - {name}")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.show()

    # 3️⃣ 输出总结果汇总表
    print("\n\n🏁 各模型交叉验证平均分对比：")
    for name, score in model_scores.items():
        print(f"{name:20s} | F1={score['F1_mean']:.4f} | ACC={score['ACC_mean']:.4f}")


    # 选择在 CV 上表现最好的模型（按 mean F1_macro）
    best_model_name = max(model_scores.items(), key=lambda kv: kv[1]['F1_mean'])[0]
    print("Best model by CV F1_macro:", best_model_name)

    # 这里对模型进行调参测试, 由于前面的代码已经可以知道针对这部分数据 RandomForest 模型的分数最好
    # 这里针对它进行优化调参, 使用 GridSearchCV 进行自动配置模型参数, 但是 GridSearchCV 需要先手动配置一下参数,
    # 这部分代码之所以存在是因为老师要求的, 直接走 else的流程也行, 只不过使用的是 RandomForest 默认参数,
    # 默认参数的得分可能没有优化后的高.
    if "RandomForest" in models:
        print("Running GridSearchCV on RandomForest (example)...")
        rf_pipe = Pipeline(steps=[("preproc", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
        # 配置 GridSearchCV 的参数
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5]
        }
        gs = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
        gs.fit(x_train, y_train)
        print("GridSearch best params:", gs.best_params_)
        print("GridSearch best score (F1_macro):", gs.best_score_)
        # 如果 GridSearch 的最佳比之前最好的还好，就选它
        if gs.best_score_ > model_scores[best_model_name]['F1_mean']:
            print("GridSearch RandomForest beats previous best. Selecting GridSearch RF as final model.")
            final_pipeline = gs.best_estimator_
            final_name = "RandomForest_GridSearch"
        else:
            # 否则用之前选出的
            final_pipeline = Pipeline(steps=[("preproc", preprocessor), ("clf", models[best_model_name])])
            final_pipeline.fit(x_train, y_train)
            final_name = best_model_name
    else:
        final_pipeline = Pipeline(steps=[("preproc", preprocessor), ("clf", models[best_model_name])])
        final_pipeline.fit(x_train, y_train)
        final_name = best_model_name

    print("Final model selected:", final_name)

    # 在保留的验证集上评估
    print("Evaluating on held-out validation set...")
    y_val_pred = final_pipeline.predict(x_val)
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
    test_pred_int = final_pipeline.predict(x_test)
    test_pred_labels = inverse_label_map(test_pred_int, label_map)

    # 生成 submission, 用 pd 库生成csv列表文件
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
    print("run time:", time.time())
    main()

