# Gunakan base image yang sama untuk konsistensi
FROM python:3.9-slim

WORKDIR /mlflow_server

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt