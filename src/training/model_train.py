# src/training/model_train.py
import argparse
import os
import ipaddress
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from joblib import dump


def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", help="Directory containing raw CSVs (CIC-IDS2018)")
    parser.add_argument("--out_dir", default="./artifacts", help="Directory to save npy artifacts")
    args = parser.parse_args()

    data_path = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load & align columns across multiple CSVs
    data_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    if not data_files:
        raise FileNotFoundError(f"No CSV files found under {data_path}")

    additional_columns = ["Src IP", "Flow ID", "Src Port", "Dst IP"]
    dfs = []
    for file in data_files:
        df = pd.read_csv(os.path.join(data_path, file), low_memory=False)

        # ensure missing extra cols exist to concat safely
        for col in additional_columns:
            if col not in df.columns:
                df[col] = np.nan
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Clean
    data = data.drop_duplicates()
    # drop rows with NaN outside the additional_columns (kept flexible)
    cols_to_check = [c for c in data.columns if c not in additional_columns]
    data = data.dropna(subset=cols_to_check)

    # IPs → ints
    data["Src IP"] = data["Src IP"].apply(ip_to_int)
    data["Dst IP"] = data["Dst IP"].apply(ip_to_int)

    # Flow ID (categorical → label)
    le_flow = LabelEncoder()
    data["Flow ID"] = le_flow.fit_transform(data["Flow ID"].astype(str))

    # Dst Port → numeric
    data["Dst Port"] = pd.to_numeric(data["Dst Port"], errors="coerce").fillna(-1)

    # Protocol (categorical → label)
    le_proto = LabelEncoder()
    data["Protocol"] = le_proto.fit_transform(data["Protocol"].astype(str))

    # Drop timestamp if present
    if "Timestamp" in data.columns:
        data = data.drop(columns=["Timestamp"])

    # Features / label
    if "Label" not in data.columns:
        raise KeyError("Column 'Label' not found in dataset.")
    features = data.columns.drop("Label")
    X = data[features].select_dtypes(include=[np.number]).copy()
    y = (data["Label"] != "Benign").astype(int)

    # Fill missing (median)
    X = X.fillna(X.median())

    # Scale → SMOTE → Split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42
    )

    # Save artifacts
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    os.makedirs("./models", exist_ok=True)
    dump(scaler, "./models/scaler.joblib")

    print(f"[OK] Artifacts saved to {out_dir}")
    print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    unique, counts = np.unique(y_bal, return_counts=True)
    print(f"Class distribution after SMOTE: {{0:{counts[0]}, 1:{counts[1]}}}")


if __name__ == "__main__":
    main()
