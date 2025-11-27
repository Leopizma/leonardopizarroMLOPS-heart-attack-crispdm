from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.s3_utils import upload_prediction_to_s3
from src.inference import predict_single
from app.schemas import HeartAttackInput


app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API para predecir riesgo de ataque cardíaco usando un modelo entrenado.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Heart Attack Prediction API is running"}


@app.post("/predict")
def predict(input_data: HeartAttackInput):
    data_dict = input_data.model_dump()

    result = predict_single(data_dict)

    full_record = {
        "input": data_dict,
        "prediction": result["prediction"],
        "probability_positive": result["probability_positive"],
    }

    # Guardar en S3 (no hace fallar la API si S3 falla)
    try:
        s3_key = upload_prediction_to_s3(full_record)
        full_record["s3_key"] = s3_key
    except Exception as e:
        print(f"Error subiendo a S3: {e}")
        full_record["s3_key"] = None

    return full_record

