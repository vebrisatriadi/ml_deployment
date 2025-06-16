import os
import logging
import ray
import mlflow
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Import Ray libraries
from ray.train.sklearn import SklearnTrainer
from ray.air.config import RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback

# Set up logging for consistency across the project.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """
    Main function to train a model using Ray Train
    and log the results to MLflow.
    """
    logging.info("Starting the training process with Ray.io...")

    # Inside a container, this will automatically create a single-node cluster.
    ray.init(address='auto', ignore_reinit_error=True)
    logging.info("Ray initialized successfully.")

    # Ray Train works best with Ray Datasets.
    # We will convert the Scikit-learn data into a Ray Dataset.
    iris_data = load_iris(as_frame=True)
    pandas_df = iris_data.frame
    # Rename the target column to match the expected format ('label')
    pandas_df.rename(columns={'target': 'label'}, inplace=True)
    
    # Convert the pandas DataFrame to a Ray Dataset
    train_dataset = ray.data.from_pandas(pandas_df)
    logging.info("Dataset prepared successfully for Ray Train.")

    # The SklearnTrainer is a wrapper that simplifies training Scikit-learn models on Ray.
    # The MLflowLoggerCallback will automatically handle logging metrics and parameters.
    trainer = SklearnTrainer(
        estimator=LogisticRegression(solver="liblinear"),
        label_column="label",
        datasets={"train": train_dataset},
        run_config=RunConfig(
            name="ray_mlflow_integration_run",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
                    experiment_name="iris_classification_ray",
                    save_artifact=True
                )
            ]
        )
    )

    # .fit() will run the training process. Logging is handled automatically by the callback.
    result = trainer.fit()
    logging.info("Ray training finished. Metrics have been logged to MLflow.")
    logging.info(f"Model checkpoint path: {result.checkpoint.path}")

    # The MLflowLoggerCallback does not automatically register the model.
    # We will do this manually by reopening the run created by the callback.
    logging.info("Registering the best model to the MLflow Model Registry...")

    # Get the run ID from the results of the training job
    mlflow_run_id = result.metrics.get("mlflow_run_id")
    if mlflow_run_id:
        # Re-open the same run to log the model
        with mlflow.start_run(run_id=mlflow_run_id):
            # Get the actual scikit-learn model from the Ray checkpoint
            sklearn_model = result.checkpoint.get_estimator()
            
            mlflow.sklearn.log_model(
                sk_model=sklearn_model,
                artifact_path="model_from_ray",
                registered_model_name="iris-classifier-ray"
            )
        logging.info("Model registered successfully with the name 'iris-classifier-ray'.")
    else:
        logging.warning("Could not find 'mlflow_run_id' in results. Skipping model registration.")
    
    # Shut down Ray
    ray.shutdown()
    logging.info("Ray shut down.")

if __name__ == "__main__":
    main()