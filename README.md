# Cara Menjalankan Proyek (End-to-End)

Dokumentasi ini berisi langkah-langkah untuk menjalankan keseluruhan sistem, mulai dari membangun lingkungan hingga mendapatkan hasil prediksi dari API.

### Prasyarat
* **Docker** & **Docker Compose** terinstall.
* **Git** terinstall.

---

### 1. Buka Proyek dan Jalankan Layanan

1.  Buka terminal Anda dan pastikan Anda berada di dalam direktori utama proyek Anda (folder yang berisi file `docker-compose.yml`).

2.  Jalankan perintah berikut untuk membangun dan memulai semua layanan secara otomatis:
    ```bash
    docker-compose up --build -d
    ```

Setelah perintah di atas selesai, layanan berikut akan berjalan:
* **MLFlow UI**: `http://localhost:5001`
* **API Service**: `http://localhost:8000`
* **MinIO Console**: `http://localhost:9001`

*(Catatan: Jika Anda menjalankan ini di mesin baru, pastikan Anda sudah melakukan `git clone` pada repositori proyek Anda terlebih dahulu untuk mendapatkan semua filenya.)*

---

### 2. Setup Penyimpanan Artefak (MinIO)

Ini adalah langkah manual satu kali untuk menyiapkan "gudang" model.

1.  Buka browser dan pergi ke **`http://localhost:9001`**.
2.  Login dengan username `minioadmin` dan password `minioadmin`.
3.  Buat sebuah *bucket* baru dengan nama persis **`mlflow`**.

---

### 3. Latih dan Daftarkan Model

Kembali ke terminal Anda dan jalankan dua perintah berikut untuk melatih model dan mendaftarkannya ke MLFlow.

```bash
# 1. Salin kode training ke dalam container server MLFlow
docker cp ./model mlflow_server:/model

# 2. Jalankan skrip training dari dalam container
docker exec -e MLFLOW_TRACKING_URI=http://mlflow_server:5001 mlflow_server bash -c "pip install -r /model/requirements.txt && python /model/train.py"
```

Pastikan di bucket `mlflow` sudah ada model artifacts yang telah dilatih.

### 4. Promosikan Model ke Produksi via UI
Beritahu sistem bahwa model yang baru dilatih siap untuk digunakan.

1.  Buka MLFlow UI di ```http://localhost:5001```.
2.  Pergi ke tab Models -> klik model `iris-classifier`.
3.  Klik pada versi model terbaru (misal: "Version 1").
4.  Klik tombol "New model registry UI" di pojok kanan atas.
5.  Di halaman baru, cari opsi `"Set alias"` atau `"Add alias"`.
6.  Ketik `production` (pastikan semua huruf kecil) sebagai nama alias dan simpan.

### Langkah 5: Hit API untuk Mendapatkan Prediksi
Sekarang sistem sudah siap sepenuhnya. Anda bisa mengujinya menggunakan curl di terminal.

Jalankan perintah di bawah ini:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
```

Hasil yang Diharapkan:
```bash
{
  "prediction_label": 0,
  "predicted_class_name": "Iris-setosa",
  "confidence_score": "87.24%"
}
```

### Train with Ray.io
Run this following command:

```bash
docker exec -e MLFLOW_TRACKING_URI=http://mlflow_server:5001 mlflow_server bash -c "pip install mlflow boto3 psycopg2-binary 'ray[air]==2.9.3' 'scikit-learn==1.3.2' 'pandas==1.5.3' && python /model/train_with_ray.py"
```