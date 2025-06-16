# Gunakan base image Python yang konsisten dengan layanan lain
FROM python:3.9-slim

WORKDIR /app/training

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "train.py"]