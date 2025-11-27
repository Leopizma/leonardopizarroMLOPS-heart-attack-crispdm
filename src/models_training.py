import os
import joblib

import mlflow
import mlflow.sklearn
import optuna

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.config import (
    DATA_PROCESSED_DIR,
    MLFLOW_EXPERIMENT_NAME,
    BEST_MODEL_PATH,
)


def load_processed_data():
    """Carga los datos procesados de data/processed/"""
    X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv"))

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    return X_train, X_test, y_train, y_test


def get_model(model_name: str, trial: optuna.trial.Trial):
   
    if model_name == "logreg":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=-1,
        )

    elif model_name == "svm":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        model = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            probability=True,
        )

    elif model_name == "rf":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return model


def objective(trial: optuna.trial.Trial, model_name: str, X_train, X_test, y_train, y_test):

    with mlflow.start_run(nested=True):
      
        mlflow.set_tag("model_name", model_name)
        model = get_model(model_name, trial)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_params(trial.params)
        mlflow.log_metric("f1_score", f1)
        return f1


def run_study_for_model(model_name: str, X_train, X_test, y_train, y_test, n_trials: int = 20):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, X_test, y_train, y_test),
        n_trials=n_trials,
    )
    return study


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_processed_data()

    best_overall = {
        "model_name": None,
        "f1": -1,
        "model_object": None,
        "trial": None,
    }

    model_names = ["logreg", "svm", "rf"]

    with mlflow.start_run(run_name="all_models_experiment"):
        for model_name in model_names:
            print(f"=== Optuna + MLflow para modelo: {model_name} ===")

            study = run_study_for_model(model_name, X_train, X_test, y_train, y_test, n_trials=20)

            best_trial = study.best_trial
            best_f1 = best_trial.value

            print(f"Mejor F1 para {model_name}: {best_f1:.4f}")
            mlflow.log_metric(f"best_f1_{model_name}", best_f1)

            final_model = get_model(model_name, best_trial)
            final_model.fit(X_train, y_train)

            if best_f1 > best_overall["f1"]:
                best_overall["model_name"] = model_name
                best_overall["f1"] = best_f1
                best_overall["model_object"] = final_model
                best_overall["trial"] = best_trial

        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        joblib.dump(best_overall["model_object"], BEST_MODEL_PATH)

        print("Mejor modelo global:")
        print("Modelo:", best_overall["model_name"])
        print("F1:", best_overall["f1"])
        print("Hiperparámetros:", best_overall["trial"].params)
        print("Modelo guardado en:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
