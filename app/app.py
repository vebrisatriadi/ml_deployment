import os
import logging
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any
from fastapi.concurrency import run_in_threadpool
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

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

Instrumentator().instrument(app).expose(app)
logging.info("Prometheus instrumentator has been set up.")

# --- Defining Custom Metrics for ML Model ---
predictions_total = Counter(
    "ml_predictions_total",
    "Total number of predictions served."
)
model_accuracy_gauge = Gauge(
    "ml_model_accuracy",
    "Current accuracy of the loaded model."
)
prediction_confidence_histogram = Histogram(
    "ml_prediction_confidence",
    "Distribution of prediction confidence scores."
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
        model_accuracy = run_data.metrics.get("accuracy", 0.0)
        logging.info(f"Registered accuracy: {model_accuracy}")

        # Prometheus: Set accuracy of gauge value ---
        model_accuracy_gauge.set(model_accuracy)

    except MlflowException as e:
        model, model_accuracy = None, 0.0
        model_accuracy_gauge.set(0.0)
        logging.warning(f"Model not found in MLflow. The application will run without a model. Error: {e}")
    except Exception as e:
        model, model_accuracy = None, 0.0
        model_accuracy_gauge.set(0.0)
        logging.error(f"A general error occurred while loading the model. Error: {e}", exc_info=True)

# --- Main Endpoint ---
@app.get("/")
def read_root():
    """
    Root endpoint that provides status information about the API and the loaded model.
    """
    model_status = "ready" if model is not None else "not ready (model not loaded)"
    accuracy_info = f"{model_accuracy:.2%}" if isinstance(model_accuracy, float) else "N/A"

    return {
        "api_status": "ok",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_STAGE,
        "model_status": model_status,
        "model_accuracy": accuracy_info
    }

# --- Catch batch request & predict the data ---
def blocking_batch_inference(model_instance, input_dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    # Function to perform batch inference on the input DataFrame.
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    try:
        # To get probabilities each class
        probabilities = model_instance._model_impl.predict_proba(input_dataframe)
        predictions = np.argmax(probabilities, axis=1)

        # Prepare results with confidence scores
        results = []
        for i, prediction_index in enumerate(predictions):
            confidence_score = float(probabilities[i, prediction_index])
            results.append({
                "prediction_index": int(prediction_index),
                "predicted_class_name": class_names[prediction_index],
                "confidence_score": confidence_score
            })
        return results

    except AttributeError:
        logging.warning("Model does not have a 'predict_proba' method. Falling back to 'predict'.")
        predictions = model_instance.predict(input_dataframe)
        results = []
        for pred in predictions:
             prediction_index = int(pred)
             results.append({
                "prediction_index": prediction_index,
                "predicted_class_name": class_names[prediction_index],
                "confidence_score": None
             })
        return results

@app.post("/predict")
async def predict(iris_batch: List[IrisInput]):
    """
    Endpoint untuk melakukan prediksi BATCH secara ASYNCHRONOUS.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready for predictions.")
    
    try:
        input_df = pd.DataFrame([item.dict() for item in iris_batch])

        # Execute prediction in a thread ppol to avoid blocking the event loop
        # Using await to keep the application responsive
        results = await run_in_threadpool(blocking_batch_inference, model, input_df)

        # To track the number of predictions and confidence scores
        for result in results:
            predictions_total.inc()
            if result.get("confidence_score") is not None:
                prediction_confidence_histogram.observe(result["confidence_score"])
        
        formatted_results = []
        for res in results:
            new_res = res.copy()
            if new_res.get("confidence_score") is not None:
                new_res["confidence_score"] = f"{new_res['confidence_score']:.2%}"
            else:
                 new_res["confidence_score"] = "not available"
            formatted_results.append(new_res)

        return {"predictions": formatted_results}

    except Exception as e:
        logging.error(f"Error during async batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# --- Endpoint Refresh ---
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