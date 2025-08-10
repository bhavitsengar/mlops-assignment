# Iris API + MLflow (with Artifact Proxy)

This project serves a simple **Iris prediction API** that loads its model from an **MLflow Tracking Server** started with `--serve-artifacts` (so clients don’t need direct access to the artifact store). It also exposes **Prometheus** metrics.

---

## Contents
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [1) Start the MLflow server](#1-start-the-mlflow-server)
- [2) Open the MLflow UI & verify artifacts](#2-open-the-mlflow-ui--verify-artifacts)
- [3) Start the Iris API container](#3-start-the-iris-api-container)
- [4) API reference & examples](#4-api-reference--examples)
- [5) Prometheus & monitoring](#5-prometheus--monitoring)

---

## Architecture
```
+--------------+         HTTP(S)         +----------------------+       scrape       +-------------+
|  Iris API    |  <--------------------> |  MLflow Tracking     | <----------------- | Prometheus  |
|  (Docker)    |        model load       |  Server (--serve-*)  |   /metrics (API)   | (optional)  |
+--------------+                         +----------------------+                   +-------------+
       ^                                         ^  artifacts
       |                                         |
       |                             Object store or local volume
       +-----------------------------------------+
```

- **MLflow server** runs with `--serve-artifacts` so the API can download models through the tracking server.
- **Iris API** reads **`MLFLOW_TRACKING_URI`** (and auth) from environment variables at runtime.
- **Prometheus** (optional) scrapes the API’s `/metrics` endpoint.

---

## Prerequisites
- Docker 20+
- (Optional) A domain or reachable host/IP for the MLflow server
- (Optional) Prometheus, Grafana
- Ports free: `5000` (MLflow), `8000` (Iris API), `9090` (Prometheus)

---

## 1) Start the MLflow server
Start MLflow with artifact proxying. You can back artifacts by a local volume or an object store (S3/GCS/ABFS). This example uses volumes.

```bash
# Create host folders
mkdir -p ~/mlflow/artifacts ~/mlflow/db

# Run MLflow server container
docker run -d \
  --name mlflow-server \
  -p 5000:5000 \
  -v ~/mlflow/artifacts:/mlflow/artifacts \
  -v ~/mlflow/db:/mlflow/db \
  -e MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/db/mlflow.db \
  -e MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts \
  ghcr.io/mlflow/mlflow:v3.1.4 \
  mlflow server \
    --backend-store-uri sqlite:///mlflow/db/mlflow.db \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts \
    --artifacts-destination /mlflow/artifacts
```

> **Tip:** For multi-host setups, an object store (e.g., `s3://my-bucket`) for `--artifacts-destination` is recommended.

---

## 2) Open the MLflow UI & verify artifacts
- Open **MLflow UI**: <http://YOUR_MLFLOW_HOST:5000>
- Create/confirm an **experiment** and **register** your model (e.g., name: `IrisModel`).
- Verify new runs use **`mlflow-artifacts:/...`** URIs (this ensures proxy downloads work). Runs created *before* enabling `--serve-artifacts` may have plain file paths and won’t proxy correctly.

Quick check with Python (optional):
```python
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://YOUR_MLFLOW_HOST:5000")
client = MlflowClient()
exp = client.get_experiment_by_name("YOUR_EXP")
print(exp.artifact_location)  # expect mlflow-artifacts:/...
```

---

## 3) Start the Iris API container
Point the container at MLflow via environment variables — no hardcoding in code.

```bash
docker run -d --name iris-api \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:5000 \
  # If your MLflow is secured, pass one of the following (optional):
  # -e MLFLOW_TRACKING_USERNAME=... \
  # -e MLFLOW_TRACKING_PASSWORD=... \
  # -e MLFLOW_TRACKING_TOKEN=... \
  # If using self-signed HTTPS:
  # -e MLFLOW_TRACKING_INSECURE_TLS=true \
  your/iris-api-image:latest
```

The API will load the model by a **Model Registry URI**, for example: `models:/IrisModel/Production` (exact name/stage are up to you).

---

## 4) API reference & examples
> Endpoint names below are the common defaults used in this project. If you changed them in code, adjust accordingly.

### Health
```
GET /health
200 OK
{"status":"ok"}
```

### Predict
- **URI:** `POST /predict`
- **Content-Type:** `application/json`
- **Request (single example, named features):**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
- **Request (batch, array of arrays):**
```json
{
  "instances": [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5]
  ]
}
```
- **Response (labels or class indices depending on your model):**
```json
{
  "predictions": ["setosa", "versicolor"]
}
```

**cURL**
```bash
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Model info (optional)
```
GET /model
200 OK
{"name":"IrisModel","stage":"Production","version":3}
```

### Prometheus metrics
```
GET /metrics
# Exposes Prometheus text format
```

---

## 5) Prometheus & monitoring
The API exposes **`/metrics`**. Point Prometheus at the API.

**Minimal `prometheus.yml`:**
```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'iris-api'
    static_configs:
      - targets: ['YOUR_IRIS_API_HOST:8000']
```

**Run Prometheus (example):**
```bash
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest
```

**Where to check Prometheus:**
- Open **Prometheus UI**: <http://YOUR_PROMETHEUS_HOST:9090>
- Go to **Status → Targets** and confirm the `iris-api` target is **UP**.
- Query examples: `process_cpu_seconds_total`, custom counters/histograms your API exports, etc.
---

If you use different ports, hostnames, or endpoint names, update the examples above accordingly.

