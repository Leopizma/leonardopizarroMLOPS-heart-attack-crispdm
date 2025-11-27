from pydantic import BaseModel


class HeartAttackInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalachh: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int