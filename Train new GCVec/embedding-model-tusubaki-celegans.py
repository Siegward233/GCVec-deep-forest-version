import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier

# === 数据加载与拼接 ===

# 训练集
protein_embeddings_train = pd.read_csv("celegans-protein-embeddings-32dim.csv").iloc[:5222]
drug_embeddings_train = pd.read_csv("celegans-drug-embeddings-32dim.csv").iloc[:5222]
labels_train = pd.read_csv("celegans-data-pre.csv")["label"][:5222]

# 测试集
protein_embeddings_test = pd.read_csv("celegans-protein-embeddings-32dim.csv").iloc[5222:].reset_index(drop=True)
drug_embeddings_test = pd.read_csv("celegans-drug-embeddings-32dim.csv").iloc[5222:].reset_index(drop=True)
labels_test = pd.read_csv("celegans-data-pre.csv")["label"][5222:]

# 拼接蛋白质和药物嵌入
X_train = np.hstack((protein_embeddings_train.values, drug_embeddings_train.values))
X_test = np.hstack((protein_embeddings_test.values, drug_embeddings_test.values))
y_train = np.array(labels_train)
y_test = np.array(labels_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# === 模型配置与训练 ===

def get_toy_config():
    return {
        "n_estimators": 2,
        "n_trees": 100,
        "max_layers": 100,
        "n_bins": 255,
        "bin_type": "percentile",
        "criterion": "gini",
        "random_state": 0,
        "n_tolerant_rounds": 10,
        "delta": 1e-5,
        "verbose": 1
    }

config = get_toy_config()
gc = CascadeForestClassifier(**config)

# 训练模型
gc.fit(X_train, y_train)

# 使用 predict_proba 作为增强特征
X_train_enc = gc.predict_proba(X_train)
X_test_enc = gc.predict_proba(X_test)

# 拼接原始特征与增强特征
X_train_combined = np.hstack((X_train, X_train_enc))
X_test_combined = np.hstack((X_test, X_test_enc))

# 保存 deepforest 模型
with open("embedding-celegans-32dim-XGB.pkl", "wb") as f:
    pickle.dump(gc, f)

# 原始模型预测
y_pred_train = gc.predict(X_train)
y_pred_test = gc.predict(X_test)
print("Train Accuracy: {:.2f}%".format(accuracy_score(y_train, y_pred_train) * 100))
print("Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_test) * 100))

# === 二阶段模型训练 ===

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train_combined, y_train)

# 保存新模型
with open("embedding-celegans-32dim-XGB-newmodel.pkl", "wb") as f:
    pickle.dump(clf, f)

# 新模型预测
y_pred_train = clf.predict(X_train_combined)
y_pred_test = clf.predict(X_test_combined)
print("Train Accuracy (new model): {:.2f}%".format(accuracy_score(y_train, y_pred_train) * 100))
print("Test Accuracy (new model): {:.2f}%".format(accuracy_score(y_test, y_pred_test) * 100))
