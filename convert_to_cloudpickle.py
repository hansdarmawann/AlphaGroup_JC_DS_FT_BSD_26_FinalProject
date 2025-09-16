import joblib
import cloudpickle

# Load your original joblib model
model_bundle = joblib.load("Model/Model_Logreg_7_fitur_Telco_Churn_joblib.pkl") 

# Save as cloudpickle version
with open("Model/Model_Logreg_Telco_Churn_cloud.pkl", "wb") as f:
    cloudpickle.dump(model_bundle, f)

print("✅ Model successfully saved as cloudpickle format.")
