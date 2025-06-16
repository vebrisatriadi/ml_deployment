import os
import logging
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

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

# --- Endpoint Prediksi ---
@app.post("/predict")
def predict(iris_input: IrisInput):
    """
    Endpoint to make predictions on new Iris data. 
    It returns the predicted class and a confidence score if available.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready for predictions.")
        
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    input_df = pd.DataFrame([iris_input.dict()])
        
    try:
        probabilities = model._model_impl.predict_proba(input_df)[0]
        
        prediction_index = int(np.argmax(probabilities))
        confidence_score = float(probabilities[prediction_index])
        predicted_class_name = class_names[prediction_index]

        return {
            "prediction_index": prediction_index,
            "predicted_class_name": predicted_class_name,
            "confidence_score": f"{confidence_score:.2%}"
        }
    except AttributeError:
        logging.warning("Model does not have a 'predict_proba' method. Falling back to 'predict'.")
        prediction = model.predict(input_df)[0]
        prediction_index = int(prediction)
        
        return {
            "prediction_index": prediction_index,
            "predicted_class_name": class_names[prediction_index],
            "confidence_score": "not available"
        }
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/refresh-model")
def refresh_model():
    """
    Endpoint to manually trigger a reload of the model from the MLflow Model Registry.
    """
    logging.info("Received request to refresh the model.")
    load_model()
    if model:
        return {"message": "Model reloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload the model.")