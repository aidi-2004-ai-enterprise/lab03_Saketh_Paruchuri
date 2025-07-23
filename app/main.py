import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import xgboost as xgb

# -------- Logging Configuration --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- Pydantic Data Contract --------
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: Sex
    island: Island

# -------- Initialize FastAPI App --------
app = FastAPI()

# -------- Load Model and Target Classes --------
try:
    model = xgb.XGBClassifier()
    model.load_model("app/data/model.json")
    logger.info("Model loaded successfully.")

    target_classes = pd.read_csv("app/data/target_classes.csv", header=None)[0].tolist()
    logger.info(f"Target classes loaded: {target_classes}")
except Exception as e:
    logger.error("Error loading model or target classes.")
    traceback.print_exc()

# -------- Helper: Prepare Input --------
def prepare_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=["sex", "island"])

    # Ensure all expected one-hot columns are present (case-sensitive!)
    for col in ["sex_Male", "sex_Female", "island_Biscoe", "island_Dream", "island_Torgersen"]:
        if col not in df.columns:
            df[col] = 0

    # Final feature order (exclude year!)
    expected_cols = [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
        "sex_Female", "sex_Male",
        "island_Biscoe", "island_Dream", "island_Torgersen"
    ]
    return df[expected_cols]

# -------- Predict Endpoint --------
@app.post("/predict")
def predict(features: PenguinFeatures):
    try:
        logger.info(f"Received input: {features}")
        input_df = prepare_input(features.dict())
        prediction_index = int(model.predict(input_df)[0])
        predicted_species = target_classes[prediction_index]
        logger.info(f"Predicted: {predicted_species}")
        return {"predicted_species": predicted_species}
    except Exception as e:
        logger.error("Prediction failed.")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

