import os
import logging
import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Mengatur logging untuk konsistensi di seluruh proyek.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure connection to the MLflow Tracking Server
# This will be picked up from environment variables in Docker Compose
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Configure connection to S3/MinIO for artifact storage
# MLflow automatically uses these if the tracking URI is set
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# Set the experiment name in the MLflow UI
mlflow.set_experiment("iris_classification")

def main():
    """Main function to train, evaluate, and log the model."""
    logging.info("Starting the training process...")

    # Load the dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info("Dataset loaded and split successfully.")

    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"MLflow run started. Run ID: {run_id}")

        # --- Parameters to track ---
        solver = "liblinear"
        
        # Log parameters
        mlflow.log_param("solver", solver)
        logging.info(f"Logged parameter 'solver': {solver}")

        # Train the model
        model = LogisticRegression(solver=solver)
        model.fit(X_train, y_train)
        logging.info("Model training complete.")

        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        logging.info(f"Logged metric 'accuracy': {accuracy:.4f}")
        
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        
        logging.info("Logging model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="iris-classifier" # Register the model in the registry
        )
        logging.info("Model trained and registered successfully.")

if __name__ == "__main__":
    main()