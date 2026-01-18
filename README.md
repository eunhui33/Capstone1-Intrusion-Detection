# Capstone1 – Intrusion Detection System for IoT Networks (Blockchain + AI)


## 📘 Overview
This project implements a **dual-layer IoT security application** that integrates:
1. **Blockchain-based Decentralized Identity (DID)** authentication for secure user access  
2. **AI-powered Intrusion Detection System (IDS)** that detects and blocks abnormal network traffic in real time  

The system ensures both **identity-level** and **network-level** security for IoT environments.  
It uses the **CIC-IDS2018 dataset** and an **MLP (Multilayer Perceptron)** model to classify traffic as *normal* or *abnormal*.
+ This dual-architecture enhances IoT resilience against spoofing, unauthorized access, and real-time intrusion attempts.

> 🏆 *Awarded “Best Undergraduate Research Paper” at KCSE 2025 (Korean Conference on Software Engineering)*

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
- Designed and implemented the end-to-end AI-based intrusion detection pipeline, focusing on real-time detection and automated response for IoT network traffic.
- Built and trained an MLP-based intrusion detection model using the CIC-IDS2018 dataset (30K samples), achieving a 93.7% F1-score through careful feature preprocessing, model tuning, and early stopping.
- Developed the data preprocessing workflow, including IP-to-integer encoding, label encoding, normalization, and SMOTE-based oversampling to address class imbalance in network traffic data.
- Implemented a real-time packet capture and inference pipeline (tshark → CICFlowMeter → FastAPI) to extract flow-level features and classify live network traffic from remote IoT devices.
- Designed evaluation and visualization modules to assess model performance, including confusion matrices, ROC-AUC, MCC, and accuracy metrics, enabling systematic validation of detection reliability.
- Deployed the intrusion prevention system on Naver Cloud Platform (NCP, AWS-equivalent) and validated real-time attack blocking using simulated DDoS and Brute Force scenarios on a Raspberry Pi IoT testbed.
- Led a 4-member team as project lead, coordinating AI model development, backend integration, milestone delivery, and weekly technical reporting to the faculty advisor.


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
-  *Best Undergraduate Research Paper — KCSE 2025* **[View Award (PDF)](./paper/KCSE2025_Best_Undergraduate_Paper_Certificate.pdf)**
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
