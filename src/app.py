from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field
import mlflow.sklearn
import numpy as np
import sqlite3
import datetime
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

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


app = FastAPI()


# --------------------------
# Prometheus metrics setup
# --------------------------
REQUEST_COUNT = Counter("predict_requests_total",
                        "Total number of prediction requests")
PREDICTION_TIME = Histogram("prediction_duration_seconds",
                            "Time spent on predictions")


# Load saved model from local directory
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "..", "models", "LogisticRegression_model")
model = mlflow.sklearn.load_model(MODEL_PATH)

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

    return {"prediction": pred}


# --------------------------
# /metrics endpoint
# --------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)