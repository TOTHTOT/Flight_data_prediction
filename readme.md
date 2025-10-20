# 航班数据集预测

---

## 🧾 一、任务总体说明

这是一个**实际数据分析（Data Analytics）项目**，是你在 Assignment 2 的数据探索之后的延续。

你要扮演一个**数据科学家（Data Scientist）**，为一家咨询公司完成一个**分类预测任务（classification task）**。

---

## 🎯 二、任务目标（Task Objectives）

你要：

1. **使用提供的数据集构建分类器（classifier）**；
2. **预测目标属性 `satisfaction`（乘客满意度）**；
3. **撰写一份报告（2000–3000 字）**；
4. **参加一个口头答辩（oral defence，约 5 分钟）**；
5. （可选）在 Kaggle 上传预测结果参加竞赛。

---

## 📦 三、提供的数据集

你会得到 3 个文件（在 Kaggle 页面下载）：

| 文件名                               | 用途                     |
| --------------------------------- | ---------------------- |
| `Assignment3-TrainingDataset.csv` | 训练集（含目标值 satisfaction） |
| `Assignment3-UnknownDataset.csv`  | 未知集（不含目标值，需要预测）        |
| `kaggle_submission_sample.csv`    | 提交样例文件（格式参考）           |

> 提交文件必须包含两列：`ID` 和 `label`。
> “label” 就是你预测的乘客满意度。

---

## 🧠 四、任务要求（What You Need to Do）

### 1️⃣ 数据处理

* 对数据进行清洗、转换或特征工程，例如：
  
  * 合并或分组某些特征；
  * 将分类变量转为数值；
  * 填补缺失值；
  * 数据标准化；
  * 解释为什么这样处理。

### 2️⃣ 构建分类器

* 使用课程中学过的 **所有分类算法（classifiers）**；
  
  > 例如：决策树、随机森林、逻辑回归、KNN、朴素贝叶斯、SVM 等等。

* 可以使用：
  
  * **KNIME**（图形化工具）；
  * 或者 **Python（推荐：scikit-learn、XGBoost 等）**；
  * 若用 Python，要在报告中说明模型原理与实现方法。

* 对比不同模型的表现（准确率、F1-score、混淆矩阵等）。

* **调参（parameter tuning）**：尝试不同参数设置，找到最佳组合。

* **验证（validation）**：用交叉验证或划分训练集 / 验证集 / 测试集来评估模型性能。

### 3️⃣ 选择最佳分类器

* 从所有尝试的模型中，挑选性能最好的一个；

* 说明：
  
  * 它的性能指标；
  * 为什么它最好；
  * 它是如何解决问题的。

### 4️⃣ （可选）Kaggle 提交

* 把你最好的模型用于未知数据集 (`Assignment3-UnknownDataset.csv`);
* 预测 `satisfaction`；
* 生成提交文件 `ID,label`；
* 上传到 Kaggle。

> ⚠️ Kaggle 评分有两部分：
> 
> * **Public Score（公开分）**：你能看到；
> * **Private Score（隐藏分）**：最终成绩时才公布。
>   所以不要只盯着公开分数调参。

---

## 🧾 五、报告要求（Report）

报告内容应包括：

| 部分                    | 要写什么                     |
| --------------------- | ------------------------ |
| **1. 问题描述**           | 说明你要解决的分类问题是什么           |
| **2. 数据预处理**          | 描述清洗、转换和特征工程步骤           |
| **3. 解决方法**           | 描述你如何建模、如何验证             |
| **4. 分类技术与结果**        | 展示你尝试的模型、参数、性能对比         |
| **5. 最佳模型说明**         | 描述最优模型的类型、性能、原理与选择理由     |
| **6. 反思（Reflection）** | 一页反思：你学到了什么、遇到的挑战、下次如何改进 |

### 字数与格式

* 约 **2000–3000 字（10–12 页）**

* 字体：Times New Roman 或 Arial，字号 11 或 12

* 文件名格式：
  
  * `fda_a3_你的学号.pdf` 或 `.doc`

---

## 🎤 六、口头答辩（Oral Defence）

* 时长约 **5 分钟**；

* 展示你的模型（KNIME workflow 或 Python 代码）；

* 回答问题，例如：
  
  * 为什么选择某个模型？
  * 数据是如何处理的？
  * 你的验证方法是什么？
  * 模型准确率是多少？

**评分标准：**

* 0, 10, 15, 20 四个等级；
* 第一次不及格可再试一次，但最高只能得 10 分。

---

## 📊 七、评分标准与占比

| 项目     | 占比       | 说明         |
| ------ | -------- | ---------- |
| 报告     | 30 分     | 模型构建、分析、反思 |
| 口头答辩   | 20 分     | 模型理解与表达    |
| **总分** | **50 分** | 占课程成绩 50%  |

---

## ⏰ 八、提交要求

* **个人作业（individual work）**
* 提交报告 PDF 或 Word
* 通过指定的提交系统（例如 LMS 或 Kaggle）
* 建议用时：**24–36 小时**

---

## ✅ 九、你要做的事情总结（Checklist）

| 步骤   | 内容                |
| ---- | ----------------- |
| 🧹 1 | 下载并探索数据集          |
| 🔧 2 | 清洗 & 特征工程         |
| 🤖 3 | 训练多种分类模型          |
| 📈 4 | 比较性能并调参           |
| 🏆 5 | 选择最优模型            |
| 📄 6 | 撰写报告（含反思）         |
| 📊 7 | （可选）Kaggle 提交预测结果 |
| 🎤 8 | 准备 5 分钟答辩，展示模型与思路 |

---

是否希望我帮你：

* ✅ 制作一份 **报告模板结构（带提示）**；
* ✅ 或者先帮你 **写出 Python 实现框架（含模型对比、交叉验证、生成 submission.csv）**？

---

# OneHot

非常好 👍，这段是针对**类别型特征（categorical columns）**的预处理。  
我们一步步拆解解释清楚：

---

## 🧩 一、背景：什么是“类别特征”

在数据集中，特征分两种：

| 类型                | 举例       | 说明              |
| ----------------- | -------- | --------------- |
| 数值特征（numeric）     | 年龄、收入、房价 | 数字型，可以直接计算      |
| 类别特征（categorical） | 性别、地区、职业 | 字符串型，不能直接用于数学计算 |

机器学习模型（例如 RandomForest、SVM 等）通常要求输入是**纯数字矩阵**，  
所以类别特征要先“转换成数字形式”，这就是这一段代码的作用。

---

## 🧮 二、第一步：填补缺失值（`SimpleImputer`）

```python
("imputer", SimpleImputer(strategy="constant", fill_value="missing"))
```

### 📘 含义

这表示：  
如果某个类别特征是空的（`NaN`），就用字符串 `"missing"` 来代替。

### 📊 举例

原始数据：

| gender | city  |
| ------ | ----- |
| male   | Tokyo |
| NaN    | Osaka |
| female | NaN   |

处理后变成：

| gender  | city    |
| ------- | ------- |
| male    | Tokyo   |
| missing | Osaka   |
| female  | missing |

> ✅ 这样模型不会因为空值（NaN）报错。

---

## 🧱 三、第二步：One-Hot 编码（`OneHotEncoder`）

```python
("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
```

### 📘 含义

`OneHotEncoder` 会把**类别型特征**转换成**独立的二进制列（0/1）**。

### 📊 举例

输入：

| gender | city  |
| ------ | ----- |
| male   | Tokyo |
| female | Osaka |

编码后变成：

| gender_male | gender_female | city_Tokyo | city_Osaka |
| ----------- | ------------- | ---------- | ---------- |
| 1           | 0             | 1          | 0          |
| 0           | 1             | 0          | 1          |

> ✅ 模型现在看到的都是数字，不再是字符串。

---

## 🧠 四、参数解释

| 参数名                       | 作用                            | 举例                                       |
| ------------------------- | ----------------------------- | ---------------------------------------- |
| `handle_unknown="ignore"` | 如果预测时出现“训练中没见过”的类别，就忽略它       | 训练时只有 `Tokyo, Osaka`，预测中出现 `Kyoto`，则不会报错 |
| `sparse_output=False`     | 返回普通的 `numpy.ndarray` 而不是稀疏矩阵 | 方便直接查看、打印数据                              |

---

## 🧰 五、Pipeline 的意义

把这两个步骤放进 `Pipeline`：

```python
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
```

等价于一个自动化流程：

1. 如果类别列有缺失 → 填 `"missing"`

2. 然后对所有类别列做 One-Hot 编码

3. 输出干净的纯数值矩阵，供模型训练使用

---

## ✅ 六、结合前面的整体结构

最终的 `ColumnTransformer`：

```python
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])
```

表示：

- 对 `numeric_cols` 使用 `numeric_transformer`（填中位数 + 标准化）

- 对 `categorical_cols` 使用 `categorical_transformer`（填 "missing" + One-Hot 编码）

---

是否希望我画一张图来帮你可视化整个 `ColumnTransformer + Pipeline` 的数据流过程？  
（能直观看出从“原始DataFrame → 预处理 → 模型输入”的变化）

---

# F1-score、混淆矩阵

非常好的问题 👍  
这两个指标 —— **F1-score（F1 分数）** 和 **混淆矩阵（Confusion Matrix）** ——  
是**分类模型最重要的评估指标**之一。  
它们能帮你清楚地了解模型到底“预测得好不好”、“错在哪儿”。

我来一步步带你理解，用通俗、直观、图形化的方式讲清楚 👇

---

## 🧠 一、首先：分类模型的预测结果有哪几种？

假设我们在做一个“乘客满意度预测”任务：  
目标是二分类：

- “satisfied”（满意）

- “neutral or dissatisfied”（不满意）

每个样本都有一个 **真实标签（actual）** 和一个 **预测标签（predicted）**。

| 实际情况  | 模型预测  | 分类结果                |
| ----- | ----- | ------------------- |
| 满意 ✅  | 满意 ✅  | True Positive (TP)  |
| 满意 ✅  | 不满意 ❌ | False Negative (FN) |
| 不满意 ❌ | 不满意 ❌ | True Negative (TN)  |
| 不满意 ❌ | 满意 ✅  | False Positive (FP) |

---

## 🧩 二、混淆矩阵（Confusion Matrix）

混淆矩阵就是把上面四种情况汇总成一个表格👇：

|                       | **预测：满意 (Positive)** | **预测：不满意 (Negative)** |
| --------------------- | -------------------- | --------------------- |
| **实际：满意 (Positive)**  | ✅ **TP**             | ❌ **FN**              |
| **实际：不满意 (Negative)** | ❌ **FP**             | ✅ **TN**              |

比如模型在 100 个乘客上预测：

| 指标      | 数值            |
| ------- | ------------- |
| TP = 50 | 模型预测满意且确实满意   |
| TN = 40 | 模型预测不满意且确实不满意 |
| FP = 5  | 模型预测满意但其实不满意  |
| FN = 5  | 模型预测不满意但其实满意  |

---

## 📊 三、常见的性能指标

根据上面的 4 个数（TP, TN, FP, FN），我们能计算很多指标。

### 1️⃣ Accuracy 准确率

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

→ 预测对的比例。

> 上例： (50 + 40) / 100 = 0.90 → 90%

⚠️ 但当类别不平衡（比如 95% 满意、5% 不满意）时，准确率容易“骗人”。

---

### 2️⃣ Precision 精确率（查准率）

```
Precision = TP / (TP + FP)
```

> 在“预测为满意”的乘客中，有多少真的满意？

例：

```
Precision = 50 / (50 + 5) = 0.909
```

---

### 3️⃣ Recall 召回率（查全率）

```
Recall = TP / (TP + FN)
```

> 所有“真正满意”的乘客中，有多少被模型找出来？

例：

```
Recall = 50 / (50 + 5) = 0.909
```

---

### 4️⃣ F1-score （F1 分数）

F1 是 **Precision 和 Recall 的调和平均数**，  
代表两者的“平衡点”：

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

当模型既要保证“预测对的样本多”（Precision高），  
又要保证“不漏掉真实的样本”（Recall高）时，F1-score 是最好的指标。

上面的例子：

```
F1 = 2 * (0.909 * 0.909) / (0.909 + 0.909) = 0.909
```

---

## 📈 四、用 scikit-learn 打印这些指标

```python
from sklearn.metrics import confusion_matrix, classification_report

# 假设 y_true 是真实标签，y_pred 是模型预测结果
cm = confusion_matrix(y_true, y_pred)
print("混淆矩阵：\n", cm)

print("\n详细分类报告：")
print(classification_report(y_true, y_pred))
```

输出：

```
混淆矩阵：
[[40  5]
 [ 5 50]]

详细分类报告：
                       precision    recall  f1-score   support
neutral or dissatisfied     0.89       0.89      0.89        45
satisfied                  0.91       0.91      0.91        55
accuracy                                   0.90       100
macro avg                  0.90       0.90      0.90       100
weighted avg               0.90       0.90      0.90       100
```

---

## 🧩 五、图形化显示混淆矩阵

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测:不满意', '预测:满意'],
            yticklabels=['实际:不满意', '实际:满意'])
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.show()
```

生成的图像：

```
        预测:不满意   预测:满意
实际:不满意   40          5
实际:满意     5          50
```

颜色越深表示数量越多。

---

## 💡 六、宏平均 F1（macro F1）

你在交叉验证中看到的：

```python
scoring="f1_macro"
```

含义是：

- 对每个类别分别计算 F1 分数（例如“满意”和“不满意”各算一次）

- 然后取**平均值**，不考虑类别数量是否均衡

> 它能公平地反映模型在所有类别上的整体表现。

---

## 🎯 七、总结一张图：

| 指标               | 计算公式                  | 含义              | 关注重点       |
| ---------------- | --------------------- | --------------- | ---------- |
| Accuracy         | (TP+TN)/(TP+TN+FP+FN) | 总体预测正确率         | 容易受不平衡数据影响 |
| Precision        | TP/(TP+FP)            | 预测为正的样本中，多少是真的正 | 控制“假阳性”    |
| Recall           | TP/(TP+FN)            | 真正的正样本中，多少被预测出来 | 控制“漏检率”    |
| F1-score         | 2 * (P*R)/(P+R)       | 精确率和召回率的平衡      | 综合表现       |
| Confusion Matrix | 表格显示 TP, FP, TN, FN   | 错误分布分析          | 模型哪里出错     |

---

## 📘 八、直觉类比（通俗版）

假设你在做“乘客满意度检测”的客服AI：

| 指标            | 类比                      |
| ------------- | ----------------------- |
| **Precision** | 你说“满意”的人里，真的满意的比例（别乱夸人） |
| **Recall**    | 所有满意的人里，你能找出多少（别漏掉太多）   |
| **F1**        | 上面两者的平衡：既别乱夸人，也别漏掉满意的乘客 |
| **混淆矩阵**      | 看看AI到底在夸错谁、漏掉谁          |

---

是否希望我帮你写一段完整的 Python 代码：  
读取模型预测结果 → 打印混淆矩阵 → 绘图 → 显示 F1/Precision/Recall 报告？  
你可以直接复制运行在你的项目里查看效果。

---

## 环境配置

```shell
# wsl 或者 powershell中运行
pip install pandas numpy scikit-learn joblib
```
