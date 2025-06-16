import os
import ray
import mlflow

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Impor pustaka Ray Train
from ray.train.sklearn import SklearnTrainer
from ray.air.config import ScalingConfig, RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback

def main():
    """
    Fungsi utama untuk melatih model menggunakan Ray Train
    dan mencatat hasilnya ke MLFlow.
    """
    print("Memulai proses training dengan Ray.io...")

    # 1. Inisialisasi atau hubungkan ke klaster Ray
    # Di dalam container, ini akan membuat klaster node tunggal secara otomatis.
    ray.init()
    print("Ray berhasil diinisialisasi.")

    # 2. Siapkan Dataset
    # Ray Train bekerja paling baik dengan Ray Dataset.
    # Kita akan mengubah data Scikit-learn menjadi Ray Dataset.
    iris_data = load_iris(as_frame=True)
    pandas_df = iris_data.frame
    # Ganti nama kolom target agar sesuai dengan format yang diharapkan
    pandas_df.rename(columns={'target': 'label'}, inplace=True)
    
    # Konversi pandas DataFrame ke Ray Dataset
    train_dataset = ray.data.from_pandas(pandas_df)
    print("Dataset berhasil disiapkan untuk Ray Train.")

    # 3. Definisikan Trainer
    # SklearnTrainer adalah wrapper yang memudahkan training model Scikit-learn di Ray.
    trainer = SklearnTrainer(
        estimator=LogisticRegression(solver="liblinear"),
        label_column="label",
        datasets={"train": train_dataset},
        # HAPUS BARIS scaling_config DARI SINI
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

    # 5. Jalankan Training
    # .fit() akan menjalankan proses training dan logging otomatis via callback.
    result = trainer.fit()
    print(f"Training dengan Ray selesai. Metrik tercatat di MLFlow.")
    print(f"Path checkpoint model: {result.checkpoint.path}")

    # 6. (Opsional tapi Best Practice) Daftarkan Model Terbaik ke Registry
    # Callback MLflow tidak mendaftarkan model secara otomatis.
    # Kita akan melakukannya secara manual setelah training selesai.
    print("Mendaftarkan model terbaik ke MLFlow Model Registry...")
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=result,
            artifact_path="model_from_ray",
            registered_model_name="iris-classifier-ray"
        )
    print("Model berhasil didaftarkan dengan nama 'iris-classifier-ray'.")
    
    # Matikan Ray
    ray.shutdown()

if __name__ == "__main__":
    main()