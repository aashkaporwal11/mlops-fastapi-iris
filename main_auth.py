import pickle
import logging
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------- LOGGING SETUP ----------------------
logging.basicConfig(
    filename="auth_app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

# ---------------------- LOAD MODEL ----------------------
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------------- FASTAPI APP ----------------------
app = FastAPI()

# ---------------------- API KEY ----------------------
API_KEY = "12345"   # You can change this

# ---------------------- INPUT SCHEMA ----------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ---------------------- GLOBAL ERROR HANDLER ----------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)}
    )


# ---------------------- HOME API ----------------------
@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Experiment 4 API Key Authentication Running"}


# ---------------------- PROTECTED PREDICT API ----------------------
@app.post("/predict")
def predict(data: IrisInput, x_api_key: str = Header(None)):

    # API Key validation
    if x_api_key != API_KEY:
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Invalid or Missing API Key")

    logger.info(f"Authorized request received: {data}")

    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(input_data)[0]

    logger.info(f"Prediction result: {prediction}")

    return {"prediction": int(prediction)}