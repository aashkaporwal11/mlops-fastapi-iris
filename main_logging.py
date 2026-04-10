import pickle
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------- LOGGING SETUP ----------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

# ---------------------- LOAD MODEL ----------------------
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------------- FASTAPI APP ----------------------
app = FastAPI()

# ---------------------- INPUT SCHEMA ----------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ---------------------- HOME API ----------------------
@app.get("/")
def home():
    logger.info("Home API accessed")
    return {"message": "FastAPI Logging Experiment 3 Running"}


# ---------------------- PREDICT API ----------------------
@app.post("/predict")
def predict(data: IrisInput):
    try:
        logger.info(f"Prediction request received: {data}")

        input_data = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = model.predict(input_data)[0]

        logger.info(f"Prediction result: {prediction}")

        return {"prediction": int(prediction)}

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {"error": "Prediction failed", "message": str(e)}


# ---------------------- GLOBAL ERROR HANDLER ----------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)}
    )