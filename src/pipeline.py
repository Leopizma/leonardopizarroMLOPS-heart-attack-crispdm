import os
from typing import List, Union

import joblib
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATASET_FILENAME,
    BEST_MODEL_PATH,
)


class HeartAttackPipeline:
    """
    Pipeline que encapsula:
    - orden de columnas
    - scaler ya entrenado
    - mejor modelo entrenado
    """

    def __init__(self, feature_names: List[str], scaler, model):
        self.feature_names = feature_names
        self.scaler = scaler
        self.model = model

    def _to_dataframe(self, X: Union[dict, pd.DataFrame, list]):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            raise ValueError("Formato de entrada no soportado")

        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en la entrada: {missing}")

        df = df[self.feature_names]
        return df

    def predict(self, X):
        df = self._to_dataframe(X)
        X_scaled = self.scaler.transform(df)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        df = self._to_dataframe(X)
        X_scaled = self.scaler.transform(df)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            raise AttributeError("El modelo no tiene método predict_proba")


def build_pipeline_from_components() -> HeartAttackPipeline:
   

    feature_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalachh", 
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    
    scaler_path = os.path.join(DATA_PROCESSED_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)

    model = joblib.load(BEST_MODEL_PATH)

    pipeline = HeartAttackPipeline(
        feature_names=feature_names,
        scaler=scaler,
        model=model,
    )

    return pipeline



def load_pipeline() -> HeartAttackPipeline:
    """
    Para compatibilidad, simplemente reconstruye el pipeline en memoria.
    """
    return build_pipeline_from_components()


if __name__ == "__main__":

    pipeline = build_pipeline_from_components()
    print("Pipeline construido correctamente con columnas:")
    print(pipeline.feature_names)
