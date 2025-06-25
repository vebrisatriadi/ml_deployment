import os
import logging
import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure connection to the MLflow Tracking Server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Configure connection to S3/MinIO for artifact storage
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
    # Ini buat inisiasi run baru dalam sebuah eksperimen
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"MLflow run started. Run ID: {run_id}")

        params = {
            "solver": "liblinear",
            "C": 0.5,
            "random_state": 42
        }
        
        mlflow.log_params(params)
        logging.info(f"Logged parameters: {params}")

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        logging.info("Model training complete.")

        # Make predictions and evaluate
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            "accuracy": accuracy,
            # "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)
        logging.info(f"Logged metrics: {metrics}")
        
        #automatically determine the input and output schema
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        
        logging.info("Logging model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="iris-classifier"
        )
        logging.info("Model trained and registered successfully.")

if __name__ == "__main__":
    main()