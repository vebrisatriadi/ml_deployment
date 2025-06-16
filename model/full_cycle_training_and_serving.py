import os
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

# --- 1. Konfigurasi Pusat ---
# Konfigurasi ini diambil dari environment variables, sama seperti di aplikasi kita.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "iris-classifier-e2e" # Menggunakan nama baru agar tidak bentrok
PRODUCTION_ALIAS = "production"

def train_and_register_model():
    """
    Fungsi ini mengenkapsulasi seluruh proses training, logging,
    versioning, dan promosi model.
    """
    print("--- Memulai Tahap Training & Registrasi ---")

    # Inisialisasi koneksi ke MLFlow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("end_to_end_workflow_experiment")

    # Memuat data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Memulai Run MLFlow
    with mlflow.start_run() as run:
        print(f"MLFlow Run Dimulai (Run ID: {run.info.run_id})")

        # Melatih model
        params = {"solver": "liblinear", "random_state": 42}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Evaluasi model
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Akurasi Model: {accuracy:.2%}")

        # Log parameter dan metrik ke MLFlow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Log model, menyimpannya ke MinIO dan mendaftarkannya ke Registry
        print(f"Mendaftarkan model dengan nama: {MODEL_NAME}")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Mempromosikan model ke alias "production" secara otomatis
        client = MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
        
        print(f"Mempromosikan Version {latest_version.version} ke alias '{PRODUCTION_ALIAS}'...")
        client.set_registered_model_alias(
            name=MODEL_NAME, alias=PRODUCTION_ALIAS, version=latest_version.version
        )
        print("--- Model Berhasil Dilatih dan Dipromosikan! ---")
    
    # Mengembalikan URI model yang siap produksi
    return f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"


@serve.deployment(ray_actor_options={"num_cpus": 1})
class MLModelServer:
    """
    Deployment Ray Serve yang akan memuat model dari MLFlow
    dan melayani permintaan prediksi.
    """
    def __init__(self, model_uri: str):
        print(f"--- Inisialisasi Model Server ---")
        print(f"Memuat model dari URI: {model_uri}")
        # Memuat model yang sudah siap produksi
        self._model = mlflow.pyfunc.load_model(model_uri)
        self._class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        print("--- Model Server Siap! ---")

    async def __call__(self, http_request):
        try:
            # Ray Serve menerima objek request seperti FastAPI
            input_data: dict = await http_request.json()
            
            # Konversi ke format yang bisa dibaca model
            df = pd.DataFrame([input_data])

            # Dapatkan probabilitas dan skor kepercayaan
            probabilities = self._model._model_impl.predict_proba(df)[0]
            prediction_label = int(np.argmax(probabilities))
            confidence_score = float(probabilities[prediction_label])
            
            return {
                "prediction_label": prediction_label,
                "predicted_class_name": self._class_names[prediction_label],
                "confidence_score": f"{confidence_score:.2%}"
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    # Inisialisasi Ray
    ray.init(address="auto")

    try:
        # 1. Latih, versioning, dan promosikan model
        production_model_uri = train_and_register_model()

        # 2. Deploy model tersebut sebagai sebuah API endpoint menggunakan Ray Serve
        deployment = MLModelServer.bind(model_uri=production_model_uri)
        
        # Jalankan server di port 8000 dengan route /predict
        serve.run(deployment, host="0.0.0.0", port=8000, route_prefix="/predict")

        input("Tekan Enter untuk menghentikan server...")

    finally:
        serve.shutdown()
        ray.shutdown()
        print("Server dan Ray telah dimatikan.")