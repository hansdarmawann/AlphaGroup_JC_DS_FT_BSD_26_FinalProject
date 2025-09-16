import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path
import sys, platform
from importlib import metadata as md

# ==== (Opsional) import komponen yang mungkin dipakai saat training ====
try:
    import sklearn
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
except Exception:
    sklearn = None

try:
    import imblearn
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    imblearn = None

try:
    import category_encoders as ce
except Exception:
    ce = None

st.set_page_config(page_title="Telco Churn", page_icon="📉")

# ====== ENV REPORT (aman) ======
def v(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return "not-installed"

with st.expander("🧪 Environment report (Cloud)"):
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

MODEL_PATH = Path("Model/Model_Logreg_Telco_Churn_cloud.pkl")

@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    try:
        with MODEL_PATH.open("rb") as f:
            bundle = cloudpickle.load(f)
        model = bundle["model"]
        sig = bundle.get("signature")
        return model, sig
    except ModuleNotFoundError as e:
        st.error(
            "❌ ModuleNotFoundError saat load model. Tambahkan modul yg hilang ke requirements "
            "dan import kelas terkait sebelum load.\n\nDetail: "
            + str(e)
        )
        st.stop()
    except AttributeError as e:
        st.error(
            "❌ AttributeError saat unpickle. Ini biasanya karena versi berbeda antara "
            "lingkungan training dan Cloud (sklearn/imbalanced-learn/numpy).\n\n"
            "🔧 Opsi:\n"
            "1) Samakan versi di `requirements.txt` dengan versi training; ATAU\n"
            "2) Re-train / re-save model memakai versi yang sama dengan yang terpasang di Cloud (lihat 'Environment report'); ATAU\n"
            "3) Simpan pakai format lebih tahan versi seperti `skops`.\n\n"
            f"Detail: {e}"
        )
        st.stop()

model, signature = load_model()
