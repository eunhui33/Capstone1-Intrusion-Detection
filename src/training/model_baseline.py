# src/training/model_baseline.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef,
    roc_auc_score, accuracy_score
)

ART = "./artifacts"
IMG = "./images"
os.makedirs(IMG, exist_ok=True)

# Load preprocessed artifacts
X = np.load(os.path.join(ART, "X_train.npy"))
y = np.load(os.path.join(ART, "y_train.npy"))
X_test = np.load(os.path.join(ART, "X_test.npy"))
y_test = np.load(os.path.join(ART, "y_test.npy"))

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP (baseline experiment)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100, warm_start=True, random_state=42)

train_accs, val_accs, test_accs = [], [], []

best_val = 0.0
patience, no_improve = 5, 0
epochs = 50

for ep in range(epochs):
    mlp.fit(X_train, y_train)
    tr = accuracy_score(y_train, mlp.predict(X_train))
    va = accuracy_score(y_val,   mlp.predict(X_val))
    te = accuracy_score(y_test,  mlp.predict(X_test))

    train_accs.append(tr); val_accs.append(va); test_accs.append(te)
    print(f"Epoch {ep+1}/{epochs} - Train {tr:.4f} | Val {va:.4f} | Test {te:.4f}")

    if va > best_val:
        best_val, no_improve = va, 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break

# Accuracy curves (saved)
plt.figure(figsize=(9,5))
plt.plot(range(1, len(train_accs)+1), train_accs, label="Train")
plt.plot(range(1, len(val_accs)+1),   val_accs,   label="Validation")
plt.plot(range(1, len(test_accs)+1),  test_accs,  label="Test")
plt.title("Accuracy per Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(IMG, "training_accuracy.png"), dpi=200)
plt.close()

# Final evaluation
y_pred = mlp.predict(X_test)
y_proba = mlp.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal","Abnormal"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout(); plt.savefig(os.path.join(IMG, "confusion_matrix.png"), dpi=200)
plt.close()

acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUC: {auc:.4f}")

# Sample check (optional)
n = min(1000, len(X_test))
idx = np.random.choice(len(X_test), n, replace=False)
pred = mlp.predict(X_test[idx])
correct = np.sum(pred == y_test[idx])
print(f"\nSample {n}: acc={correct/n:.4f}")
