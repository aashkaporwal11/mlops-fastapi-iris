import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load saved model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Create FastAPI app
app = FastAPI()

# Request Body Schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Response Schema
class PredictionOutput(BaseModel):
    prediction: int


@app.get("/")
def home():
    return {"message": "FastAPI Iris Model is Running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(input_data)[0]

    return {"prediction": int(prediction)}