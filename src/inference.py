from typing import Dict, Any

import pandas as pd

from src.pipeline import load_pipeline

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline


def predict_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza una predicción para un solo paciente.
    """
    pipeline = get_pipeline()

    df = pd.DataFrame([input_data])

    pred = pipeline.predict(df)[0]

    result = {"prediction": int(pred)}

    try:
        proba = pipeline.predict_proba(df)[0][1]
        result["probability_positive"] = float(proba)
    except Exception:
        result["probability_positive"] = None

    return result
