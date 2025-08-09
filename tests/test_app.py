from fastapi.testclient import TestClient
from src.app import app
import os

os.environ["SKIP_MODEL_LOAD"] = "1"


class DummyModel:
    def predict(self, X):
        # return one label per row
        return [1] * len(X)


# Inject dummy
app.model = DummyModel()

client = TestClient(app)


def test_predict_success():
    # Sample valid input
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=input_data)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)


def test_predict_missing_field():
    # Missing "petal_width"
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post("/predict", json=input_data)

    # Unprocessable Entity (validation error)
    assert response.status_code == 422
