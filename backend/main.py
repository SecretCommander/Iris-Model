from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('backend/Irismodel.pkl')

class InputData(BaseModel):
    SepalLengthCm: float	
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.post("/predict")
def predict(data: InputData):
    try:
        input_data = np.array([[data.SepalLengthCm, data.SepalWidthCm, data.PetalLengthCm, data.PetalWidthCm]])
        
        label_mapping = {
            0: "Iris-setosa",
            1: "Iris-versicolor",
            2: "Iris-virginica"
        }

        prediction = model.predict(input_data)
        return {"prediction": label_mapping.get(prediction[0],'Unknown Class')}
    except Exception as e:
        return {"error": str(e)}

# , '"confidence": model.predict_proba(input_data).max()']
