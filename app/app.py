import os
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

class IrisInput(BaseModel):
    sepal_length: float; sepal_width: float; petal_length: float; petal_width: float

app = FastAPI(title="Iris Classifier API", version="1.0.0")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

MODEL_NAME = "iris-classifier"
# MODEL_NAME = "iris-classifier-ray"
MODEL_STAGE = "production"
model = None
model_accuracy = None

@app.on_event("startup")
def load_model():
    global model, model_accuracy
    model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}" 
    print(f"Mencoba memuat model dari URI: {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model '{MODEL_NAME}' dengan alias '{MODEL_STAGE}' berhasil dimuat.")
        print("Mencoba mengambil metrik akurasi...")

        client = MlflowClient()
        model_version_details = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE)
        run_id = model_version_details.run_id
        print(f"Mengambil metrik dari Run ID: {run_id}")

        run_data = client.get_run(run_id).data
        model_accuracy = run_data.metrics.get("accuracy", "tidak tersedia")
        print(f"Akurasi yang tercatat: {model_accuracy}")

    except MlflowException as e:
        model, model_accuracy = None, None; print(f"WARNING: Model tidak ditemukan di MLFlow. Aplikasi akan berjalan tanpa model."); print(f"Detail error: {e}")
    except Exception as e:
        model, model_accuracy = None, None; print(f"WARNING: Terjadi error umum saat memuat model. Aplikasi akan berjalan tanpa model."); print(f"Detail error: {e}")

@app.get("/")
def read_root():
    # ... (Fungsi read_root tidak perlu diubah, biarkan seperti sebelumnya)
    model_status = "siap" if model is not None else "belum siap (model tidak termuat)"
    accuracy_info = "N/A"
    if model_accuracy is not None and isinstance(model_accuracy, float):
        accuracy_info = f"{model_accuracy:.2%}"
    return {
        "status_api": "ok", "model_name": MODEL_NAME, "model_alias": MODEL_STAGE,
        "model_status": model_status, "model_accuracy": accuracy_info
    }

@app.post("/predict")
def predict(iris_input: IrisInput):
    """Endpoint untuk melakukan prediksi beserta skor kepercayaannya."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap...")
        
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    input_data = pd.DataFrame([iris_input.dict()])
        
    try:
        probabilities = model._model_impl.predict_proba(input_data)[0]
        
        prediction_label = int(np.argmax(probabilities))
        confidence_score = float(probabilities[prediction_label])
        predicted_class_name = class_names[prediction_label]

        return {
            "prediction_label": prediction_label,
            "predicted_class_name": predicted_class_name,
            "confidence_score": f"{confidence_score:.2%}"
        }
    except AttributeError:
        print("Peringatan: Model tidak memiliki metode predict_proba. Kembali ke predict biasa.")
        prediction = model.predict(input_data)[0]
        prediction_label = int(prediction)
        
        return {
            "prediction_label": prediction_label,
            "predicted_class_name": class_names[prediction_label],
            "confidence_score": "tidak tersedia"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.post("/refresh-model")
def refresh_model():
    load_model()
    if model: return {"message": "Model berhasil dimuat ulang."}
    else: raise HTTPException(status_code=500, detail="Gagal memuat ulang model.")