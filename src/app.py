from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.sklearn
import numpy as np
import sqlite3
import datetime


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

# Load saved model from local directory
model = mlflow.sklearn.load_model("/app/models/LogisticRegression_model")

@app.post("/predict")
def predict(data: IrisInput):
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
