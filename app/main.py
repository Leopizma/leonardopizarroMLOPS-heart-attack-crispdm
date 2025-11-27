from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.inference import predict_single
from src.s3_utils import save_prediction_to_s3
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
   
    try:
        
        data_dict = input_data.dict()

      
        prediction_result = predict_single(data_dict)

        
        record = {
            "input": data_dict,
            "prediction": prediction_result["prediction"],
            "probability_positive": prediction_result["probability_positive"],
        }

        
        try:
            save_prediction_to_s3(record)
        except Exception as e:
            print(f"Error guardando en S3 (no fatal): {e}")

        return record

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
