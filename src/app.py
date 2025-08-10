from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import mlflow.sklearn
import numpy as np
import pandas as pd
import sqlite3
import datetime
import os
import logging
import sys
from sklearn.base import clone
from src.helper import normalize_df
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from prometheus_client import Counter, Histogram
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


# -------------------------------
# CONFIG
# -------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://146.56.165.194:5000"
)
MODEL_NAME = "Iris Production Model"
DATA_PATH = "data/iris.csv"
os.environ["MLFLOW_MODEL_STAGE"] = "Auto"
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"


# Input validation schema using Pydantic
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10,
                                description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, lt=10,
                               description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, lt=10,
                                description="Petal length in cm")
    petal_width: float = Field(..., gt=0, lt=10,
                               description="Petal width in cm")


class IrisRow(IrisInput):
    target: int = Field(..., ge=0, le=2,
                        description="target class (0, 1, or 2)")


class TrainRequest(BaseModel):
    data: list[IrisRow]


app = FastAPI()


# --------------------------
# Prometheus metrics setup
# --------------------------
REQUEST_COUNT = Counter("predict_requests_total",
                        "Total number of prediction requests")
PREDICTION_TIME = Histogram("prediction_duration_seconds",
                            "Time spent on predictions")


# Connect to remote MLflow tracking server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# Load production model by name
model = None
if not SKIP_MODEL_LOAD:
    # Use MLflow client to fetch the latest model version
    model = mlflow.sklearn.load_model("models:/"+MODEL_NAME+"/None")
else:
    # If skipping model load, use a dummy model for testing
    class DummyModel:
        def predict(self, X):
            return [1] * len(X)

    model = DummyModel()

# Configure logger
logger = logging.getLogger("iris_api")
logger.setLevel(logging.INFO)

# Log to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Format: timestamp - level - message
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.propagate = False  # Prevent duplicate logs


# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
@PREDICTION_TIME.time()
def predict(data: IrisInput):
    REQUEST_COUNT.inc()

    input_array = np.array([[  # input must be 2D
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    pred = int(model.predict(input_array)[0])

    # Set up SQLite logging
    conn = sqlite3.connect("logs.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            input TEXT,
            output TEXT
        )
    """)
    conn.commit()

    # Logging
    log_entry = (str(datetime.datetime.now()), str(data.dict()), str(pred))
    c.execute("INSERT INTO logs VALUES (?, ?, ?)", log_entry)
    conn.commit()

    # Log to stdout
    logger.info(f"Prediction made | Input: {data.dict()} | Output: {pred}")

    return {"prediction": pred}


# --------------------------
# /metrics endpoint
# --------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -------------------------------
# /train
# -------------------------------
@app.post("/train")
def train_on_new_data(request: TrainRequest):
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"{DATA_PATH} not found")
        global model
        # Load existing and new data
        existing_df = pd.read_csv(DATA_PATH)
        existing_df = normalize_df(existing_df)
        new_df = pd.DataFrame([row.dict() for row in request.data])
        new_df = normalize_df(new_df)

        # Combine and deduplicate
        combined_df = pd.concat([existing_df, new_df],
                                ignore_index=True).drop_duplicates()
        combined_df.to_csv(DATA_PATH, index=False)

        # Handle missing values (if any)
        combined_df = combined_df.dropna()

        # Encode the target column as categorical (if needed)
        combined_df['target'] = combined_df['target'].astype('category')

        # Feature scaling (standardization)

        features = combined_df.columns[:-1]  # all columns except 'target'
        scaler = StandardScaler()
        combined_df[features] = scaler.fit_transform(combined_df[features])

        # Prepare data
        X = combined_df[features]
        y = combined_df['target'].cat.codes

        new_model = new_model = clone(model)
        new_model.fit(X, y)

        preds = new_model.predict(X)
        acc = accuracy_score(y, preds)
        recall = recall_score(y, preds, average="macro")
        f1 = f1_score(y, preds, average="macro")

        # Log to MLflow
        mlflow.set_experiment("iris_production_retraining")
        with mlflow.start_run():
            mlflow.log_params(new_model.get_params())
            mlflow.log_metrics({"accuracy": acc, "recall": recall,
                                "f1_score": f1})
            mlflow.sklearn.log_model(new_model, artifact_path="model")

            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                MODEL_NAME
            )

        # Reload the model to ensure it's updated
        if not SKIP_MODEL_LOAD:
            model = mlflow.sklearn.load_model(
                "models:/" + MODEL_NAME + "/None"
            )

        return {
            "message": "Model retrained and logged to MLflow",
            "accuracy": acc,
            "rows": len(combined_df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
