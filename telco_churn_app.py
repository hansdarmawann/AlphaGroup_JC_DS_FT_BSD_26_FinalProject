# telco_churn_app.py
import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path
import sys, platform, os
from importlib import metadata as md

# ===== ENV REPORT =====
st.set_page_config(page_title="Telco Churn", page_icon="📉")

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

# ==== (Opsional) import komponen training supaya unpickle mengenali kelas/objek ====
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

# ===== PATHS =====
APP_DIR = Path(__file__).parent.resolve()
MODEL_DIR = APP_DIR / "Model"
MODEL_PATH = MODEL_DIR / "Model_Logreg_Telco_Churn_cloud.pkl"

st.caption(f"🔎 Looking for model at: `{MODEL_PATH}`")

# ===== FILE EXPLORER (bantu cek LFS/pemanggilan path) =====
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

# ===== MODEL LOADER (tanpa hard-stop) =====
@st.cache_resource(show_spinner=True)
def try_load_model(model_path: Path):
    """
    Return: (model_or_none, info_dict)
    info_dict = {
      "exists": bool, "size": int|None, "is_lfs_pointer": bool,
      "error": str|None, "path": str
    }
    """
    info = {"exists": model_path.exists(), "size": None, "is_lfs_pointer": False,
            "error": None, "path": str(model_path)}
    if not info["exists"]:
        info["error"] = f"Model file NOT FOUND at {model_path}"
        return None, info

    try:
        size = model_path.stat().st_size
        info["size"] = size

        # Deteksi Git LFS pointer
        if size is not None and size < 500:
            try:
                head = model_path.read_text(errors="ignore")
            except Exception:
                head = ""
            if "git-lfs.github.com/spec/v1" in head:
                info["is_lfs_pointer"] = True
                info["error"] = (
                    "Model file looks like a Git LFS POINTER, not the real binary.\n"
                    "→ Aktifkan Git LFS di repo & pastikan file besar terunduh di Cloud.\n"
                    "   - `git lfs install`\n"
                    "   - `git lfs track \"Model/*.pkl\"`\n"
                    "   - commit `.gitattributes` & file `.pkl`\n"
                    "   - push ulang, lalu redeploy app"
                )
                return None, info

        with model_path.open("rb") as f:
            bundle = cloudpickle.load(f)
        if not isinstance(bundle, dict) or "model" not in bundle:
            info["error"] = "Bundle tidak valid (tidak ada key 'model')."
            return None, info

        return bundle["model"], info

    except ModuleNotFoundError as e:
        info["error"] = (
            "ModuleNotFoundError saat unpickle. Tambahkan modul hilang ke requirements.txt "
            "dan IMPORT kelas terkait sebelum load.\nDetail: " + str(e)
        )
        return None, info
    except AttributeError as e:
        info["error"] = (
            "AttributeError saat unpickle. Biasanya karena versi paket berbeda antara "
            "training vs Cloud (sklearn/imbalanced-learn/numpy). Samakan versi atau re-save model.\n"
            "Detail: " + str(e)
        )
        return None, info
    except Exception as e:
        info["error"] = f"Gagal load model: {e}"
        return None, info

model, model_info = try_load_model(MODEL_PATH)

with st.expander("📦 Model diagnostics (detail)", expanded=True if model is None else False):
    st.write(model_info)

if model is None:
    # Tampilkan pesan jelas, tapi JANGAN stop — biar UI tetap muncul untuk debugging
    st.error("❌ Model belum berhasil di-load. Lihat panel '📦 Model diagnostics' & '📁 Model folder listing' di atas.")
else:
    st.success("✅ Model loaded successfully.")

# ===== UI =====
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
