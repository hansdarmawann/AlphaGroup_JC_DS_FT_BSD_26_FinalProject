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

# Direct download URL (RAW GitHub). Bisa diganti via Secrets: st.secrets["MODEL_URL"]
MODEL_URL = (
    st.secrets.get("MODEL_URL", "").strip()
    or "https://raw.githubusercontent.com/hansdarmawann/AlphaGroup_JC_DS_FT_BSD_26_FinalProject/main/Model/Model_Logreg_Telco_Churn_cloud.pkl"
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

# (opsional) import komponen training agar unpickle mengenali kelas/objek
try:
    import sklearn
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
except Exception:
    pass
try:
    import imblearn
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    pass
try:
    import category_encoders as ce
except Exception:
    pass

# ========= PATHS =========
APP_DIR = Path(__file__).parent.resolve()
MODEL_DIR = APP_DIR / "Model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "Model_Logreg_Telco_Churn_cloud.pkl"

st.caption(f"🔎 Looking for model at: `{MODEL_PATH}`")

# ========= UTILS =========
def list_dir(p: Path, depth: int = 1):
    rows = []
    try:
        if p.exists():
            for item in sorted(p.iterdir()):
                size = None
                try:
                    if item.is_file():
                        size = item.stat().st_size
                except Exception:
                    pass
                rows.append({
                    "path": str(item.relative_to(APP_DIR)),
                    "type": "file" if item.is_file() else "dir",
                    "size_bytes": size
                })
                if item.is_dir() and depth > 1:
                    for sub in sorted(item.iterdir()):
                        size2 = None
                        try:
                            if sub.is_file(): size2 = sub.stat().st_size
                        except Exception:
                            pass
                        rows.append({
                            "path": str(sub.relative_to(APP_DIR)),
                            "type": "file" if sub.is_file() else "dir",
                            "size_bytes": size2
                        })
        else:
            rows.append({"path": str(p), "type": "missing", "size_bytes": None})
    except Exception as e:
        rows.append({"path": f"ERROR reading {p}", "type": "error", "size_bytes": str(e)})
    return pd.DataFrame(rows)

with st.expander("🗂️ App root listing", expanded=False):
    st.dataframe(list_dir(APP_DIR, depth=1), use_container_width=True)
with st.expander("📁 Model folder listing", expanded=True):
    st.dataframe(list_dir(MODEL_DIR, depth=1), use_container_width=True)

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
    info = {
        "path": str(path),
        "exists_before": path.exists(),
        "size_before": (path.stat().st_size if path.exists() else None),
        "is_lfs_pointer_before": is_lfs_pointer(path) if path.exists() else None,
        "download_attempted": False,
        "download_info": None,
        "exists_after": None,
        "size_after": None,
        "is_lfs_pointer_after": None,
        "checksum_ok": None,
        "error": None,
    }

    need_download = (not path.exists()) or is_lfs_pointer(path)
    if need_download:
        if not url or "PUT_YOUR_MODEL_URL_HERE" in url:
            info["error"] = (
                "Model file missing atau LFS pointer, dan MODEL_URL belum valid. "
                "Set di Secrets atau konstanta MODEL_URL."
            )
        else:
            info["download_attempted"] = True
            dl = download_model(url, path)
            info["download_info"] = dl
            if dl.get("error"):
                info["error"] = dl["error"]

    info["exists_after"] = path.exists()
    info["size_after"] = (path.stat().st_size if path.exists() else None)
    info["is_lfs_pointer_after"] = is_lfs_pointer(path) if path.exists() else None

    if info["exists_after"] and checksum:
        try:
            calc = sha256_of(path)
            info["checksum_ok"] = (calc.lower() == checksum.lower())
            if not info["checksum_ok"]:
                info["error"] = f"Checksum mismatch. expected={checksum} got={calc}"
        except Exception as e:
            info["error"] = f"Checksum calc failed: {e}"

    return info

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
    except ModuleNotFoundError as e:
        return None, (
            "ModuleNotFoundError saat unpickle. Tambahkan modul ke requirements.txt "
            f"dan import kelas terkait. Detail: {e}"
        )
    except AttributeError as e:
        return None, (
            "AttributeError saat unpickle (versi paket berbeda?). "
            f"Detail: {e}"
        )
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
