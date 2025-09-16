# telco_churn_app.py
import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path
import sys, platform, hashlib
from importlib import metadata as md

# ========= CONFIG =========
st.set_page_config(page_title="Telco Churn", page_icon="📉")

# Direct download URL (Google Drive)
MODEL_URL = (
    st.secrets.get("MODEL_URL", "").strip()
    or "https://drive.google.com/uc?export=download&id=1QlqQs2fGOV0stQ8VwPY5dus7WNW4Vkug"
)
MODEL_SHA256 = st.secrets.get("MODEL_SHA256", "").strip()  # opsional

# ========= ENV REPORT =========
def v(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:
        return "not-installed"

with st.expander("🧪 Environment report (Cloud)", expanded=False):
    st.write({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "streamlit": v("streamlit"),
        "numpy": v("numpy"),
        "pandas": v("pandas"),
        "scikit-learn": v("scikit-learn"),
        "imbalanced-learn": v("imbalanced-learn"),
        "cloudpickle": v("cloudpickle"),
        "scipy": v("scipy"),
        "category-encoders": v("category-encoders"),
    })

# ========= PATHS =========
APP_DIR = Path(__file__).parent.resolve()
MODEL_DIR = APP_DIR / "Model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "Model_Logreg_Telco_Churn_cloud.pkl"

st.caption(f"🔎 Looking for model at: `{MODEL_PATH}`")

# ========= UTILS =========
def is_lfs_pointer(path: Path) -> bool:
    try:
        if not path.exists(): return False
        if path.stat().st_size >= 500:
            return False
        head = path.read_text(errors="ignore")
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

@st.cache_resource(show_spinner=True)
def download_model(url: str, dest: Path) -> dict:
    info = {"download_url": url, "saved_to": str(dest), "bytes": 0, "error": None}
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = 0
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            info["bytes"] = total
        return info
    except Exception as e:
        info["error"] = f"Download failed: {e}"
        return info

def ensure_model_available(path: Path, url: str, checksum: str = "") -> dict:
    need_download = (not path.exists()) or is_lfs_pointer(path)
    if need_download:
        dl = download_model(url, path)
        if dl.get("error"):
            return {"error": dl["error"], "download_info": dl}
    if not path.exists():
        return {"error": "Model file not found after download"}
    if checksum:
        calc = sha256_of(path)
        if calc.lower() != checksum.lower():
            return {"error": f"Checksum mismatch. expected={checksum} got={calc}"}
    return {"ok": True, "path": str(path), "size": path.stat().st_size}

availability = ensure_model_available(MODEL_PATH, MODEL_URL, MODEL_SHA256)
with st.expander("📦 Model availability & download status", expanded=True):
    st.write(availability)

# ========= LOAD MODEL =========
@st.cache_resource(show_spinner=True)
def try_load_model(model_path: Path):
    if not model_path.exists():
        return None, f"Model not found at {model_path}"
    if is_lfs_pointer(model_path):
        return None, "File is a Git LFS pointer, not the actual model binary."
    try:
        with model_path.open("rb") as f:
            bundle = cloudpickle.load(f)
        if not isinstance(bundle, dict) or "model" not in bundle:
            return None, "Bundle invalid (missing key 'model')."
        return bundle["model"], None
    except Exception as e:
        return None, f"Gagal load model: {e}"

model, load_err = try_load_model(MODEL_PATH)

if load_err:
    st.error("❌ Model belum berhasil di-load.")
    with st.expander("🔍 Load error detail", expanded=True):
        st.code(load_err, language="text")
else:
    st.success("✅ Model loaded successfully.")

# ========= UI FORM =========
st.title("📉 Telco Customer Churn Prediction")
st.header("🔍 Enter Customer Information")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
with col2:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2000.0, step=50.0)
with col3:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

st.divider()
predict_disabled = (model is None)
if st.button("Predict Churn", disabled=predict_disabled):
    try:
        row = {
            'gender': str(gender),
            'SeniorCitizen': int(1 if senior == "Yes" else 0),
            'Partner': str(partner),
            'Dependents': str(dependents),
            'tenure': float(tenure),
            'MonthlyCharges': float(monthly_charges),
            'TotalCharges': float(total_charges),
            'PhoneService': str(phone_service),
            'MultipleLines': str(multiple_lines),
            'InternetService': str(internet_service),
            'OnlineSecurity': str(online_security),
            'OnlineBackup': str(online_backup),
            'DeviceProtection': str(device_protection),
            'TechSupport': str(tech_support),
            'StreamingTV': str(streaming_tv),
            'StreamingMovies': str(streaming_movies),
            'Contract': str(contract),
            'PaperlessBilling': str(paperless_billing),
            'PaymentMethod': str(payment_method),
        }
        input_data = pd.DataFrame([row])

        with st.expander("📋 Debug Input", expanded=False):
            st.write(input_data)
            st.write(input_data.dtypes)

        pred = model.predict(input_data)[0]
        proba = float(model.predict_proba(input_data)[0][1])

        if int(pred) == 1:
            st.error(f"⚠️ Customer Likely to Churn (Probability: {proba:.2%})")
        else:
            st.success(f"✅ Customer Likely to Stay (Probability: {proba:.2%})")
    except Exception as e:
        st.exception(f"❌ Prediction failed: {e}")
elif predict_disabled:
    st.warning("🔒 Tombol Predict nonaktif karena model belum berhasil di-load.")
