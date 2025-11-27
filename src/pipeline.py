import os
import joblib
import pandas as pd
from typing import List, Union

from src.config import (
    DATA_PROCESSED_DIR,
    BEST_MODEL_PATH,
    PIPELINE_PATH,
)


class HeartAttackPipeline:
    
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
            raise ValueError("Formato de entrada malo")

        
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


def create_and_save_pipeline():
    
    # Cargar columnas de entrenamiento
    X_train_path = os.path.join(DATA_PROCESSED_DIR, "X_train.csv")
    X_train = pd.read_csv(X_train_path)
    feature_names = list(X_train.columns)

    # Cargar scaler y modelo
    scaler_path = os.path.join(DATA_PROCESSED_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)

    model = joblib.load(BEST_MODEL_PATH)

    # Crear pipeline
    pipeline = HeartAttackPipeline(
        feature_names=feature_names,
        scaler=scaler,
        model=model,
    )

    # Guardar pipeline
    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)

    print(f"Pipeline guardado en: {PIPELINE_PATH}")


def load_pipeline() -> HeartAttackPipeline:
    """Carga el pipeline desde disco."""
    return joblib.load(PIPELINE_PATH)


if __name__ == "__main__":
    create_and_save_pipeline()
