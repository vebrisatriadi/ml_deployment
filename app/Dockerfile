# Gunakan base image resmi Python
FROM python:3.9-slim

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin file requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua kode aplikasi
COPY . .

# Expose port yang akan digunakan oleh Uvicorn
EXPOSE 8000

# Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]