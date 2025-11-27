import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, DATASET_FILENAME

def load_raw_data():
    """Carga el archivo."""
    path = os.path.join(DATA_RAW_DIR, DATASET_FILENAME)
    df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    """limpieza."""
    # El dataset no tiene nulos críticos, pero se deja la función por si se necesita.
    df = df.drop_duplicates()
    return df

def split_features_target(df, target_col="target"):
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_test_splitting(X, y, test_size=0.2, random_state=42):
    """train-test"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    """StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar scaler
    scaler_path = os.path.join(DATA_PROCESSED_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled

def save_processed_data(X_train, X_test, y_train, y_test):
    """Guarda"""

    pd.DataFrame(X_train).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "X_train.csv"), index=False
    )

    pd.DataFrame(X_test).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "X_test.csv"), index=False
    )

    pd.DataFrame(y_train).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "y_train.csv"), index=False
    )

    pd.DataFrame(y_test).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "y_test.csv"), index=False
    )
