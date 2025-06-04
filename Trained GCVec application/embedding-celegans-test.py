
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from deepforest import CascadeForestClassifier

# === 数据加载与拼接 ===

# 训练集
protein_embeddings_train = pd.read_csv("celegans-protein-embeddings-32dim.csv").iloc[:5222]
length_train = len(protein_embeddings_train["mean_ci_2"])
protein_embeddings_train = [protein_embeddings_train.loc[i].values for i in range(length_train)]

drug_embeddings_train = pd.read_csv("celegans-drug-embeddings-32dim.csv").iloc[:5222]
drug_embeddings_train = [drug_embeddings_train.loc[i].values for i in range(length_train)]

all_embeddings_train = np.hstack((protein_embeddings_train, drug_embeddings_train))
all_embeddings_train = [all_embeddings_train[i].reshape(64,) for i in range(length_train)]
labels_train = pd.read_csv("celegans-data-pre.csv")["label"][:5222]

# 测试集
protein_embeddings_test = pd.read_csv("celegans-protein-embeddings-32dim.csv").iloc[5222:]
protein_embeddings_test = protein_embeddings_test.reset_index(drop=True)
length_test = len(protein_embeddings_test["mean_ci_2"])
protein_embeddings_test = [protein_embeddings_test.loc[i].values for i in range(length_test)]

drug_embeddings_test = pd.read_csv("celegans-drug-embeddings-32dim.csv").iloc[5222:]
drug_embeddings_test = drug_embeddings_test.reset_index(drop=True)
drug_embeddings_test = [drug_embeddings_test.loc[i].values for i in range(length_test)]

all_embeddings_test = np.hstack((protein_embeddings_test, drug_embeddings_test))
all_embeddings_test = [all_embeddings_test[i].reshape(64,) for i in range(length_test)]
labels_test = list(pd.read_csv("celegans-data-pre.csv")["label"][5222:])

X_train = np.array(all_embeddings_train)
y_train = np.array(labels_train)
X_test = np.array(all_embeddings_test)
y_test = np.array(labels_test)

# 加载 deepforest 模型
with open("embedding-celegans-32dim-XGB.pkl", "rb") as f:
    gc = pickle.load(f)

# 预测与评估 deepforest 模型
y_pred_train = gc.predict(X_train)
y_pred_train_proba = gc.predict_proba(X_train)
pd.DataFrame(y_pred_train_proba).to_csv("y-pred-train-celegans-proba-32dim.csv", index=False)

acc_train = accuracy_score(y_train, y_pred_train)
auc_train = roc_auc_score(y_train, y_pred_train_proba[:, 1])

y_pred_test = gc.predict(X_test)
y_pred_test_proba = gc.predict_proba(X_test)
pd.DataFrame(y_pred_test).to_csv("y-pred-test-celegans-32dim.csv", index=False)
pd.DataFrame(y_pred_test_proba).to_csv("y-pred-test-celegans-proba-32dim.csv", index=False)

acc_test = accuracy_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_pred_test_proba[:, 1])
precision_test = precision_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)
matthews_corrcoef_test = matthews_corrcoef(y_test, y_pred_test)

bins = np.array([0, 0.5, 1])
tn, fp, fn, tp = np.histogram2d(y_test, y_pred_test, bins=bins)[0].flatten()
SP = tn / (tn + fp)
SE = tp / (tp + fn)

print("Test Accuracy of GcForest (train) = {:.2f} %".format(acc_train * 100))
print("Test Accuracy of GcForest (test) = {:.2f} %".format(acc_test * 100))
print("AUC (train) = {:.4f}".format(auc_train))
print("AUC (test) = {:.4f}".format(auc_test))
print("Precision (test) = {:.4f}".format(precision_test))
print("Recall (SE) (test) = {:.4f}".format(SE))
print("F1 Score (test) = {:.4f}".format(f1_score_test))
print("MCC (test) = {:.4f}".format(matthews_corrcoef_test))
print("Specificity (SP) (test) = {:.4f}".format(SP))

# 特征增强与二阶段模型评估
X_train_enc_test = gc.predict_proba(X_train)
X_test_enc = gc.predict_proba(X_test)

X_train_origin = X_train
X_test_origin = X_test

X_train_enc_test = np.hstack((X_train_origin, X_train_enc_test))
X_test_enc = np.hstack((X_test_origin, X_test_enc))

print("X_train_enc_test.shape={}, X_test_enc.shape={}".format(X_train_enc_test.shape, X_test_enc.shape))

# 加载二阶段模型
with open("embedding-celegans-32dim-XGB-newmodel.pkl", "rb") as f:
    clf = pickle.load(f)

# 预测与评估二阶段模型
y_pred_train = clf.predict(X_train_enc_test)
y_pred_train_proba = clf.predict_proba(X_train_enc_test)
pd.DataFrame(y_pred_train_proba).to_csv("y-pred-train-celegans-proba-32dim-newmodel.csv", index=False)

acc_train = accuracy_score(y_train, y_pred_train)
auc_train = roc_auc_score(y_train, y_pred_train_proba[:, 1])

y_pred_test = clf.predict(X_test_enc)
y_pred_test_proba = clf.predict_proba(X_test_enc)
pd.DataFrame(y_pred_test).to_csv("y-pred-test-celegans-32dim-newmodel.csv", index=False)
pd.DataFrame(y_pred_test_proba).to_csv("y-pred-test-celegans-proba-32dim-newmodel.csv", index=False)

acc_test = accuracy_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_pred_test_proba[:, 1])
precision_test_new = precision_score(y_test, y_pred_test)
f1_score_test_new = f1_score(y_test, y_pred_test)
matthews_corrcoef_test_new = matthews_corrcoef(y_test, y_pred_test)

tn, fp, fn, tp = np.histogram2d(y_test, y_pred_test, bins=bins)[0].flatten()
SP = tn / (tn + fp)
SE = tp / (tp + fn)

print("Test Accuracy of newmodel (train) = {:.2f} %".format(acc_train * 100))
print("Test Accuracy of newmodel (test) = {:.2f} %".format(acc_test * 100))
print("AUC of newmodel (train) = {:.4f}".format(auc_train))
print("AUC of newmodel (test) = {:.4f}".format(auc_test))
print("Precision of newmodel (test) = {:.4f}".format(precision_test_new))
print("Recall (SE) of newmodel (test) = {:.4f}".format(SE))
print("F1 Score of newmodel (test) = {:.4f}".format(f1_score_test_new))
print("MCC of newmodel (test) = {:.4f}".format(matthews_corrcoef_test_new))
print("Specificity (SP) of newmodel (test) = {:.4f}".format(SP))
