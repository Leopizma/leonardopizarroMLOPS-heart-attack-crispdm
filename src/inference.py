from typing import Dict, Any

import pandas as pd

from src.pipeline import load_pipeline

# Cargar pipeline
_pipeline = load_pipeline()


def predict_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza una predicción para un solo paciente.

    input_data: diccionario con las features del paciente.
    Devuelve: dict con predicción y probabilidad (si existe).
    """
    df = pd.DataFrame([input_data])

    pred = _pipeline.predict(df)[0]

    result = {"prediction": int(pred)}

    try:
        proba = _pipeline.predict_proba(df)[0][1] 
        result["probability_positive"] = float(proba)
    except Exception:
        result["probability_positive"] = None

    return result
