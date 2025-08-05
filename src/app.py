from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()
model = model = mlflow.sklearn.load_model("/app/models/LogisticRegression_model")

@app.post("/predict")
def predict(data: IrisInput):
    x = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    pred = model.predict(x)[0]
    return {"prediction": int(pred)}
