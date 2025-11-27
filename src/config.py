import os

# carpeta raiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas de datos
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# dataset
DATASET_FILENAME = "heart.csv"
DATASET_PATH = os.path.join(DATA_RAW_DIR, DATASET_FILENAME)

# MLflow
MLFLOW_EXPERIMENT_NAME = "heart_attack_experiment"

# modelos
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PIPELINE_PATH = os.path.join(MODELS_DIR, "heart_attack_pipeline.joblib")
os.makedirs(MODELS_DIR, exist_ok=True)
