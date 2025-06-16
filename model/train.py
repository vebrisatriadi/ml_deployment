import os
import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Atur koneksi ke MLFlow Tracking Server
# Variabel ini akan diambil dari environment variables di Docker Compose
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Atur koneksi ke S3/MinIO untuk menyimpan artifact
# MLFlow akan otomatis menggunakan ini jika tracking URI sudah di-set
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# Nama eksperimen di UI MLFlow
mlflow.set_experiment("iris_classification")

def main():
    """Fungsi utama untuk melatih, mengevaluasi, dan mencatat model."""
    print("Memulai proses training...")

    # Memuat dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Memulai MLFlow run
    with mlflow.start_run() as run:
        # Parameter yang ingin dilacak
        solver = "liblinear"
        
        # Log parameter
        mlflow.log_param("solver", solver)
        print(f"Run ID: {run.info.run_id}")

        # Melatih model
        model = LogisticRegression(solver=solver)
        model.fit(X_train, y_train)

        # Melakukan prediksi dan evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrik
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy}")
        
        # Mendefinisikan signature model (input & output)
        # Ini adalah best practice agar MLFlow tahu skema data model kita
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        
        # Log model ke MLFlow
        # Model akan disimpan di MinIO pada path: s3://mlflow/artifacts/...
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="iris-classifier" # Daftarkan model ke registry
        )
        print("Model berhasil dilatih dan didaftarkan.")

if __name__ == "__main__":
    main()