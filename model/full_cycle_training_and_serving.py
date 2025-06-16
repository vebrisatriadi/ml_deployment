import os
import logging
import ray
import mlflow
import numpy as np
import pandas as pd
from ray import serve
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

# This configuration is loaded from environment variables, just like in our app.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "iris-classifier-e2e" # Using a new name to avoid conflicts
MODEL_STAGE = "production"       # Standardized name for the production alias

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_and_register_model():
    """
    This function encapsulates the entire process of training, logging,
    versioning, and promoting the model.
    """
    logging.info("--- Starting Training & Registration Phase ---")

    # Initialize connection to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("end_to_end_workflow_experiment")

    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start an MLflow Run
    with mlflow.start_run() as run:
        logging.info(f"MLflow Run Started (Run ID: {run.info.run_id})")

        # Train the model
        params = {"solver": "liblinear", "random_state": 42}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"Model Accuracy: {accuracy:.2%}")

        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model, saving it to MinIO and registering it in the Registry
        logging.info(f"Registering model with name: {MODEL_NAME}")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Instead of querying all versions, we directly use the version
        # from the `model_info` object we just received. This is more reliable.
        client = MlflowClient()
        latest_version = model_info.version
        
        logging.info(f"Promoting Version {latest_version} to alias '{MODEL_STAGE}'...")
        client.set_registered_model_alias(
            name=MODEL_NAME, alias=MODEL_STAGE, version=latest_version
        )
        logging.info("--- Model Successfully Trained and Promoted! ---")
    
    # Return the URI for the production-ready model
    return f"models:/{MODEL_NAME}@{MODEL_STAGE}"


@serve.deployment(ray_actor_options={"num_cpus": 1})
class MLModelServer:
    """
    A Ray Serve deployment that loads a model from MLflow
    and serves prediction requests.
    """
    def __init__(self, model_uri: str):
        logging.info("--- Initializing Model Server ---")
        logging.info(f"Loading model from URI: {model_uri}")
        # Load the production-ready model
        self._model = mlflow.pyfunc.load_model(model_uri)
        self._class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        logging.info("--- Model Server is Ready! ---")

    async def __call__(self, http_request):
        try:
            # Ray Serve receives a request object similar to FastAPI
            input_data: dict = await http_request.json()
            
            # Convert to a format the model can read
            df = pd.DataFrame([input_data])

            # Get probabilities and confidence score
            probabilities = self._model._model_impl.predict_proba(df)[0]
            prediction_label = int(np.argmax(probabilities))
            confidence_score = float(probabilities[prediction_label])
            
            return {
                "prediction_label": prediction_label,
                "predicted_class_name": self._class_names[prediction_label],
                "confidence_score": f"{confidence_score:.2%}"
            }
        except Exception as e:
            # --- Improvement 2: Enhanced Server-Side Error Logging ---
            # Log the full error on the server for easier debugging.
            logging.error(f"An error occurred during prediction: {e}", exc_info=True)
            # Return a generic error to the client.
            return {"error": "An internal error occurred during prediction."}


if __name__ == "__main__":
    # Initialize Ray
    ray.init(address="auto", ignore_reinit_error=True)

    try:
        # 1. Train, version, and promote the model
        production_model_uri = train_and_register_model()

        # 2. Deploy that model as an API endpoint using Ray Serve
        deployment = MLModelServer.bind(model_uri=production_model_uri)
        
        # Run the server on port 8000 with the route /predict
        serve.run(deployment, host="0.0.0.0", port=8000, route_prefix="/predict")

        input("Server is running. Press Enter to stop the server...")

    finally:
        serve.shutdown()
        ray.shutdown()
        logging.info("Server and Ray have been shut down.")