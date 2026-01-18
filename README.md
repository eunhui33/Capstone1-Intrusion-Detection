# Capstone1 – Intrusion Detection System for IoT Networks (Blockchain + AI)


## 📘 Overview
This project implements a **dual-layer IoT security application** that integrates:
1. **Blockchain-based Decentralized Identity (DID)** authentication for secure user access  
2. **AI-powered Intrusion Detection System (IDS)** that detects and blocks abnormal network traffic in real time  

The system ensures both **identity-level** and **network-level** security for IoT environments.  
It uses the **CIC-IDS2018 dataset** and an **MLP (Multilayer Perceptron)** model to classify traffic as *normal* or *abnormal*.
+ This dual-architecture enhances IoT resilience against spoofing, unauthorized access, and real-time intrusion attempts.

> 🏆 *Awarded “Best Undergraduate Research Paper” at KCSE 2025*

---

## ⚙️ Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.10 |
| **AI / ML** | Scikit-learn (MLPClassifier), CatBoost |
| **EDA / Preprocessing** | Pandas, NumPy, Matplotlib, Seaborn |
| **Modeling / Deployment** | scikit-learn (MLP), imbalanced-learn (SMOTE), FastAPI, Uvicorn |
| **Infrastructure** | CUDA, cuDNN, CICFlowMeter, tcpdump, Wireshark |
| **Dataset** | CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity) |

---

## 💻 My Contributions
- Built the **MLP training pipeline** with structured logging, early stopping, and real-time performance tracking  
- Designed **data preprocessing workflow**:  
  → IP to integer encoding, label encoding, normalization, and SMOTE oversampling  
- Developed **evaluation and visualization module** (confusion matrix, ROC-AUC, MCC, Accuracy)  
- Implemented **FastAPI backend** to perform real-time packet capture and classification using `CatBoost`  
- Collaborated with three teammates, leading the AI model development and backend integration.
- Led a 4-member team.


---

## 📊 Results
| Model | Accuracy | AUC | MCC |
|--------|-----------|---------|-----|
| **MLP (my model)** | 0.9581 | 0.9714 | 0.8740 |
| CatBoost (team baseline) | 0.9370 | 0.9808 | 0.8240 |

> ✅ The MLP achieved stable detection performance with minimal false positives,  
> effectively identifying abnormal IoT traffic patterns in real time.

---

## 🏅 Recognition & Documents
-  *Best Undergraduate Research Paper — KCSE 2025*  
  - **[View Certificate (PDF)](./paper/KCSE2025_Best_Undergraduate_Paper_Certificate.pdf)**
-  Paper (Korean): **[KCSE 2025 IoT IDS Paper (PDF)](./paper/KCSE2025_IoT_IDS_Paper_KR.pdf)**


---

## 🧩 Repository Structure
```
Capstone1-Intrusion-Detection/
├── README.md
├── src/
│ ├── training/
│ │ ├── model_train.py      # MLP training pipeline (main)
│ │ └── model_baseline.py   # initial experiment version
│ ├── inference/
│ │ └── app_fastapi.py      # real-time FastAPI backend
├── paper/
│   ├── KCSE2025_Best_Undergraduate_Paper_Certificate.pdf
│   └── KCSE2025_IoT_IDS_Paper_KR.pdf
├── images/                       # confusion matrix / training curves
└── requirements.txt
```
