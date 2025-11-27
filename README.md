# leonardopizarroMLOPS-heart-attack-crispdm
MLOPS/taller3/2025-2

# Heart Attack Prediction

## 1. Entendimiento del Negocio

El objetivo de negocio es construir un sistema que pueda predecir el riesgo de ataque al corazon a partir de variables dadas por la clinica.  
Con esto podemos:
- Priorizar pacientes de alto riesgo
- Apoyar decisiones de prevencion
- Optimizar recursos medicos.

### Objetivo analitico

Entrenar modelos de Machine Learning capaces de clasificar si un paciente tiene riesgo de ataque al corazon

## 2. Entendimiento de los Datos

<img width="308" height="408" alt="image" src="https://github.com/user-attachments/assets/6da46692-1196-40e1-a3e6-d6f2630e08ec" />
<img width="1386" height="253" alt="image" src="https://github.com/user-attachments/assets/cac915a2-5128-4b84-9aac-8331288c6a7f" />
<img width="482" height="359" alt="image" src="https://github.com/user-attachments/assets/1ee873af-bd86-4588-b10f-90244096ac65" />
<img width="525" height="429" alt="image" src="https://github.com/user-attachments/assets/57c4dc4a-009f-4195-8056-3583c6804db2" />

## 3. Preparación de Datos
Todo el preprocesamiento está en:
src/data_prep.py
notebooks/02_preparation_and_modeling.ipynb

## 4. Modelación
<img width="1903" height="1033" alt="docs_mlflow_all_models_run" src="https://github.com/user-attachments/assets/68ebe17f-4e2b-4d59-bf6d-04ed03e0e095" />

## 5. Evaluación

| Modelo              | Mejor F1     |
| ------------------- | ------------ |
| Logistic Regression | 0.8307       |
| SVM                 | 0.8484       |
| Random Forest       | **0.8571** |



## 6. Despliegue (MLOps)

- Creación de un **pipeline** completo que reciba datos crudos y devuelva una predicción.
- Implementación de un servicio con **FastAPI** exponiendo un endpoint `/predict`.
- Registro de las predicciones en un bucket de **S3**.
- Despliegue del servicio en una instancia **EC2**.
- Configuración de un **daemon con systemd** para mantener la app siempre activa.
- Prueba del endpoint usando `curl` desde la máquina local.

---

## 7. Estructura del proyecto

```text
data/
 ├─ raw/            # Dataset original descargado de Kaggle
 └─ processed/      # Datos transformados/listos para modelar

notebooks/          # Notebooks de EDA, preparación, modelado, evaluación

src/
 ├─ config.py       # Rutas, nombres de experimento, configuración general
 ├─ data_prep.py    # Funciones de carga y preparación de datos
 ├─ models_training.py  # Entrenamiento de modelos + Optuna + MLflow
 ├─ pipeline.py     # Pipeline final de inferencia
 ├─ inference.py    # Funciones para usar el pipeline en producción
 └─ s3_utils.py     # Funciones para guardar info en S3

app/
 └─ main.py         # Aplicación FastAPI con el endpoint de predicción

deployment/
 ├─ ec2_setup.md    # Pasos realizados en la instancia EC2
 ├─ service_fastapi.service  # Archivo de systemd para levantar la app
 └─ curl_examples.md # Ejemplos de llamadas al endpoint usando curl

docs/
 ├─ crispdm_report.md       # Documento describiendo cada fase CRISP-DM
 └─ aws_screenshots/        # Capturas de AWS, systemctl, curl, etc.
