# src/inference/app_fastapi.py
import os
import subprocess
from fastapi import FastAPI, HTTPException
import uvicorn

# Env-configurable settings
MODEL_PATH = os.getenv("MODEL_PATH", "./models/catboost_model.cbm")
PCAP_DIR = os.getenv("PCAP_DIR", "./pcaps")
CICFLOWMETER_BIN = os.getenv("CICFLOWMETER_BIN", "cicflowmeter")
SSH_KEY = os.getenv("SSH_KEY", "~/.ssh/id_rsa")
SSH_HOST = os.getenv("SSH_HOST", "user@your-remote-host")
CAPTURE_PORT = os.getenv("CAPTURE_PORT", "9000")
PACKET_COUNT = os.getenv("PACKET_COUNT", "20000")
CONNECT_TIMEOUT = os.getenv("CONNECT_TIMEOUT", "45")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.05"))  # 5%

os.makedirs(PCAP_DIR, exist_ok=True)

# (Pseudo) model holder
class Detector:
    def __init__(self, path):
        self.path = path
        self.loaded = False

    def load(self):
        # TODO: load real model; here we just simulate load check
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Model not found: {self.path}")
        self.loaded = True

    def predict_batch(self, csv_path):
        # TODO: parse CSV and return predictions list like ["정상", "비정상", ...]
        # For demo purpose: pretend most are normal and a few are abnormal.
        return ["정상"] * 95 + ["비정상"] * 5

detector = Detector(MODEL_PATH)

app = FastAPI(title="IoT IDS Demo", version="1.0.0")


@app.on_event("startup")
def _load_model():
    try:
        detector.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector.loaded}


@app.post("/capture-and-detect")
def capture_and_detect():
    unique_name = "capture"
    pcap_file = os.path.join(PCAP_DIR, f"{unique_name}.pcap")
    csv_file = os.path.join(PCAP_DIR, f"{unique_name}.csv")

    # Remote tcpdump → local pcap
    capture_cmd = (
        f"ssh -i {SSH_KEY} {SSH_HOST} -o ConnectTimeout={CONNECT_TIMEOUT} "
        f"\"/usr/sbin/tcpdump -i any -U -c {PACKET_COUNT} -w - port {CAPTURE_PORT}\" | "
        f"tshark -r - -F pcap -w {pcap_file}"
    )

    try:
        subprocess.run(capture_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Capture failed: {e}")

    # pcap → csv (CICFlowMeter)
    try:
        subprocess.run(f"{CICFLOWMETER_BIN} -f {pcap_file} -c {csv_file}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"CICFlowMeter failed: {e}")

    # Predict
    preds = detector.predict_batch(csv_file)
    abnormal_ratio = preds.count("비정상") / max(1, len(preds))
    decision = "abnormal" if abnormal_ratio >= ALERT_THRESHOLD else "normal"

    return {
        "total_flows": len(preds),
        "abnormal_ratio": round(abnormal_ratio, 4),
        "threshold": ALERT_THRESHOLD,
        "decision": decision
    }


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
