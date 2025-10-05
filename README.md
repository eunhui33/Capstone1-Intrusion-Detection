# 🔐 Capstone1 – Intrusion Detection System for IoT Networks (Blockchain + AI)

## 📘 Overview
This project implements a **dual-layer IoT security application** that integrates:
1. **Blockchain-based Decentralized Identity (DID)** authentication for secure user access  
2. **AI-powered Intrusion Detection System (IDS)** that detects and blocks abnormal network traffic in real time  

The system ensures both **identity-level** and **network-level** security for IoT environments.  
It uses the **CIC-IDS2018 dataset** and an **MLP (Multilayer Perceptron)** model to classify traffic as *normal* or *abnormal*.

> 🏆 *Awarded “Best Undergraduate Research Paper” at KCSE 2025*

---

## ⚙️ Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.x |
| **AI / ML** | Scikit-learn (MLPClassifier), CatBoost |
| **Data Processing** | Pandas, NumPy, StandardScaler, SMOTE |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Backend / Deployment** | FastAPI, Uvicorn |
| **Infrastructure** | CUDA, cuDNN, CICFlowMeter, tcpdump, Wireshark |
| **Dataset** | CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity) |

---

## 💻 My Contributions
- Built the **MLP training pipeline** with structured logging, early stopping, and real-time performance tracking  
- Designed **data preprocessing workflow**:  
  → IP to integer encoding, label encoding, normalization, and SMOTE oversampling  
- Developed **evaluation and visualization module** (confusion matrix, ROC-AUC, MCC, Accuracy)  
- Implemented **FastAPI backend** to perform real-time packet capture and classification using `CatBoost`  
- Configured **GPU acceleration (CUDA/cuDNN)** for faster training and inference  

---

## 📊 Results
| Model | Accuracy | ROC-AUC | MCC |
|--------|-----------|---------|-----|
| **MLP (my model)** | 93.7% | 0.9714 | 0.91 |
| CatBoost (team baseline) | 95.8% | 0.9808 | — |

> ✅ The MLP achieved stable detection performance with minimal false positives,  
> effectively identifying abnormal IoT traffic patterns in real time.

---

## 🚀 How to Run
> Note: This repository is for research demonstration purposes.  
> The dataset (CIC-IDS2018) is not included due to licensing and size restrictions.

To reproduce model training (using local CSVs):
```bash
# 1️⃣ Clone the repository
git clone https://github.com/eunhui33/Capstone1-Intrusion-Detection.git
cd Capstone1-Intrusion-Detection

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Train model (requires dataset under /data)
python src/training/model_train.py

---

🏅 Recognition

Best Undergraduate Research Paper – KCSE 2025

Developed as part of a 4-member capstone team

---

## 🧩 Repository Structure
Capstone1-Intrusion-Detection/
├── README.md
├── src/
│ ├── training/
│ │ ├── model_train.py # MLP training pipeline (main)
│ │ └── model_baseline.py # initial experiment version
│ ├── inference/
│ │ └── app_fastapi.py # real-time FastAPI backend
├── paper/
│ └── KCSE2025_IoT_IDS.pdf
├── images/
│ ├── confusion_matrix.png
│ ├── training_history.png
│ └── architecture.png
└── requirements.txt
