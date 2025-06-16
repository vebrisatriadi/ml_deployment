import os
import logging
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from typing import List
import json
from typing import List, Dict, Any # -> Impor tipe tambahan
from fastapi.concurrency import run_in_threadpool # -> Impor utilitas async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI(
    title="Iris Classifier API",
    description="An API to classify Iris flowers and demonstrate ML model deployment.",
    version="1.0.0"
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

MODEL_NAME = "iris-classifier"
MODEL_STAGE = "production"
model = None
model_accuracy = None

@app.on_event("startup")
def load_model():
    """
    Loads the machine learning model from the MLflow Model Registry
    during the application's startup.
    """
    global model, model_accuracy
    model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
    logging.info(f"Attempting to load model from URI: {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"Model '{MODEL_NAME}@{MODEL_STAGE}' loaded successfully.")
        
        logging.info("Fetching accuracy metric...")
        client = MlflowClient()
        model_version_details = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE)
        run_id = model_version_details.run_id
        logging.info(f"Fetching metrics from Run ID: {run_id}")
        
        run_data = client.get_run(run_id).data
        model_accuracy = run_data.metrics.get("accuracy", "not available")
        logging.info(f"Registered accuracy: {model_accuracy}")

    except MlflowException as e:
        model, model_accuracy = None, "not available"
        logging.warning(f"Model not found in MLflow. The application will run without a model. Error: {e}")
    except Exception as e:
        model, model_accuracy = None, "not available"
        logging.error(f"A general error occurred while loading the model. Error: {e}", exc_info=True)

# --- Endpoint Utama ---
@app.get("/")
def read_root():
    """
    Root endpoint that provides status information about the API and the loaded model.
    """
    model_status = "ready" if model is not None else "not ready (model not loaded)"
    accuracy_info = "N/A"
    if isinstance(model_accuracy, float):
        accuracy_info = f"{model_accuracy:.2%}"
    elif model_accuracy:
        accuracy_info = model_accuracy

    return {
        "api_status": "ok",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_STAGE,
        "model_status": model_status,
        "model_accuracy": accuracy_info
    }

# --- Endpoint Predict ---
def blocking_batch_inference(model_instance, input_dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Fungsi SINKRONUS yang berisi logika inferensi CPU-bound.
    Fungsi ini akan kita jalankan di thread terpisah agar tidak memblokir server.
    """
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    try:
        # Model scikit-learn secara alami mendukung prediksi batch pada DataFrame multi-baris
        probabilities = model_instance._model_impl.predict_proba(input_dataframe)
        predictions = np.argmax(probabilities, axis=1)

        results = []
        for i, prediction_index in enumerate(predictions):
            confidence_score = float(probabilities[i, prediction_index])
            results.append({
                "prediction_index": int(prediction_index),
                "predicted_class_name": class_names[prediction_index],
                "confidence_score": f"{confidence_score:.2%}"
            })
        return results

    except AttributeError:
        logging.warning("Model does not have a 'predict_proba' method. Falling back to 'predict'.")
        # Fallback jika model tidak memiliki predict_proba
        predictions = model_instance.predict(input_dataframe)
        results = []
        for pred in predictions:
             prediction_index = int(pred)
             results.append({
                "prediction_index": prediction_index,
                "predicted_class_name": class_names[prediction_index],
                "confidence_score": "not available"
             })
        return results

@app.post("/predict")
async def predict(iris_batch: List[IrisInput]): # -> Endpoint sekarang ASYNC
    """
    Endpoint untuk melakukan prediksi BATCH secara ASYNCHRONOUS.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready for predictions.")
    
    try:
        # Konversi input Pydantic ke DataFrame (ini proses yang cepat, aman di main thread)
        input_df = pd.DataFrame([item.dict() for item in iris_batch])
        
        # Jalankan fungsi inferensi yang berat (blocking) di thread pool
        # `await` akan menunggu hasilnya tanpa memblokir event loop utama.
        results = await run_in_threadpool(blocking_batch_inference, model, input_df)

        return {"predictions": results}

    except Exception as e:
        logging.error(f"Error during async batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


# --- Endpoint Refresh (Tidak ada perubahan) ---
@app.post("/refresh-model")
def refresh_model():
    """
    Endpoint untuk memicu pemuatan ulang model secara manual dari MLflow Model Registry.
    """
    logging.info("Received request to refresh the model.")
    load_model()
    if model:
        return {"message": "Model reloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload the model.")