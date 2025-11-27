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

Mejor modelo Random Forest 
<img width="498" height="455" alt="image" src="https://github.com/user-attachments/assets/a0e100ae-9df1-4687-80ce-e3a535f7b068" />

## 6. Despliegue (MLOps)
src/pipeline.py
<img width="569" height="320" alt="image" src="https://github.com/user-attachments/assets/c9307668-a5bb-4b4c-b194-b2e37efa0d82" />

Despliegue en EC2: 
git clone <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nano .env #Keys
uvicorn app.main:app --host 0.0.0.0 --port 8000

systemctl status:
<img width="1894" height="997" alt="status fastapi" src="https://github.com/user-attachments/assets/1d85cd3a-68cf-451f-bd00-6be53a8be283" />
el servidor funcionando:
<img width="1912" height="970" alt="JSON" src="https://github.com/user-attachments/assets/0246e795-4439-4419-8a49-80ace64a9a19" />

