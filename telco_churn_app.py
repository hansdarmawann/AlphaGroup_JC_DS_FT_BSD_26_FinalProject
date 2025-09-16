import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path

# >>> Import semua lib yg mungkin dipakai saat training (aman walau tdk dipakai persis)
# Jika tak dipakai, bisa dihapus; kalau dipakai, ini membantu unpickle menemukan kelasnya.
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
    imblearn = None

try:
    import category_encoders as ce  # kalau kamu pakai TargetEncoder dkk
except Exception:
    ce = None

st.set_page_config(page_title="Telco Churn", page_icon="📉")

# ====== ENV REPORT ======
with st.expander("🧪 Environment report (Cloud)"):
    def safe_ver(mod, name):
        try:
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "not-imported"
    st.write({
        "python": st.runtime.scriptrunner.get_script_run_ctx().streamlit_version,  # versi Streamlit
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cloudpickle": getattr(cloudpickle, "__version__", "unknown"),
        "sklearn": safe_ver(sklearn, "sklearn") if 'sklearn' in globals() else "not-imported",
        "imblearn": safe_ver(imblearn, "imblearn") if imblearn else "not-imported",
        "category_encoders": safe_ver(ce, "category_encoders") if ce else "not-imported"
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
            "dan import kelas terkait sebelum load.\n\n"
            f"Detail: {e}"
        )
        st.stop()
    except AttributeError as e:
        # Ini tipikal mismatch versi sklearn/imblearn → attribute hilang/berubah
        st.error(
            "❌ AttributeError saat unpickle. Sangat mungkin karena perbedaan versi "
            "scikit-learn/imbalanced-learn/numpy antara training vs Cloud.\n\n"
            "🔧 Opsi fix cepat:\n"
            "1) Samakan versi lib di `requirements.txt` dgn yang dipakai saat training; atau\n"
            "2) Re-export / simpan ulang model di environment yg VERSINYA sama seperti Cloud; atau\n"
            "3) Gunakan format penyimpanan yg lebih tahan versi (mis. `skops`).\n\n"
            f"Detail: {e!s}"
        )
        st.stop()

model, signature = load_model()
