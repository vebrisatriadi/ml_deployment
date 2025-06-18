# Running the Project: End-to-End

This doc walks you through everything you need to get the whole system up and running, from building the environment to getting predictions from the API.

### Prerequisite
* **Docker** & **Docker Compose** installed.
* **Git** installed..

---

### Step 1: Fire Up the Services

1.  Pop open your terminal and make sure you're in the project's main directory (the one with the `docker-compose.yml` file).

2.  Run the following command to build and start all the services in the background:
    ```bash
    docker-compose up --build -d
    ```

Once that's done, you'll have a few services running:
* **MLFlow UI**: `http://localhost:5001`
* **API Service**: `http://localhost:8000`
* **MinIO Console**: `http://localhost:9001`
* **Prometheus UI**: `http://localhost:9090`
* **Grafana UI**: `http://localhost:3000`

*(Quick note: If you're running this on a new machine, make sure you've already `git clone` this repo to get all the project files!)*

### Step 2: Set Up the Artifact Store (MinIO)

This is a one-time manual step to get our model "warehouse" ready.

1.  Head over to **`http://localhost:9001`** in your browser.
2.  Log in with the username `minioadmin` and password `minioadmin`.
3.  Create a new bucket and name it exactly `mlflow`.

### Step 3: Train & Register the Model

Alright, back to your terminal. Run these two commands to train our first model and get it registered in MLflow.

```bash
# Build and train your model
docker-compose up --build model_trainer
```

You can double-check that the model artifacts have appeared in your `mlflow` bucket in MinIO.


### Step 4: Promote the Model to Production via the UI
Now, let's tell our system that this new model is ready for prime time.

1.  Go to the MLflow UI at ```http://localhost:5001```.
2.  Navigate to the Models tab -> click on the `iris-classifier` model.
3.  Click on the latest version (e.g., "Version 1").
4.  Click the "`New model registry UI`" button in the top right corner.
5.  On the new page, look for the "`Set alias`" or "`Add alias`" option.
6.  Type `production` (all lowercase) as the alias name and save it.

### Step 5: Hit the API for a Prediction!
And that's it! The system is fully armed and operational. You can test it out with a simple `curl` command in your terminal.

Run the command below:

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

Expected Result:
```bash
{
  "prediction_label": 0,
  "predicted_class_name": "Iris-setosa",
  "confidence_score": "87.24%"
}
```

## Additional Info

### Update Your Model
If you have any update about your model, just rebuild `model_trainer` container and restart `fastapi_app` container.

```bash
# Rebuild model_trainer container
docker-compose up --build model_trainer

# Restart fastapi_app container
docker-compose restart fastapi_app
```

### Check your API dan models performance in Grafana
1. Open your Grafana UI `http://localhost:3000`
2. Login with user:`admin` and password:`admin`
2. Go to General, click `ML API Monitoring`

### Scale Up your ML App
#### Vertical Scale
On `app/Dockefile`:
```bash
# Replace
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# with
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Then, rebuild your container
```bash
docker-compose up --build model_trainer
```

#### Horizontal Scale
On root directory, you'll find `docker-compose.yml` file, open the file.

You'll see `---- HORIZONTAL SCALLING ----` section in this configuration. **Uncomment** it.

After that, comment section
```bash
fastapi_app:
  build: ./app
  container_name: fastapi_app
  ports:
    - "8000:8000"
  environment:
    - MLFLOW_TRACKING_URI=http://mlflow_server:5001
    - AWS_ACCESS_KEY_ID=minioadmin
    - AWS_SECRET_ACCESS_KEY=minioadmin
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  depends_on:
    - mlflow_server
```

Then rebuild your container
```bash
# Stop your container with 
docker compose down

# Rebuild and run with scale 
docker-compose up --build --scale fastapi_app=3 -d
```


### Train with Ray.io
If you want to try out the distributed training script using Ray, run this single command instead of the commands in Step 3.

```bash
docker-compose down mlflow_server

docker exec -e MLFLOW_TRACKING_URI=http://mlflow_server:5001 mlflow_server bash -c "pip install mlflow boto3 psycopg2-binary 'ray[air]==2.9.3' 'scikit-learn==1.3.2' 'pandas==1.5.3' && python /model/train_with_ray.py"
```