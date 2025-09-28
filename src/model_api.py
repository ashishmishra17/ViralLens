from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Load cleaned data and train model (for demo purposes, train on startup)
df = pd.read_csv('../placement_cleaned.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Optionally, save model for reuse
# joblib.dump(model, 'rf_model.joblib')

class InputData(BaseModel):
    data: dict  # Accepts a dictionary of feature values for prediction

@app.post('/predict')
async def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.data])
    pred = model.predict(input_df)[0]
    return {'prediction': int(pred)}

# Sample test (run this FastAPI app, then use below curl or Python request):
# curl example:
# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"data": {"feature1": value1, "feature2": value2, ...}}'

# Python request example:
# import requests
# response = requests.post("http://127.0.0.1:8000/predict", json={"data": {"feature1": value1, "feature2": value2, ...}})
# print(response.json())
