import joblib
import cloudpickle
import sys
import os
from pydantic import BaseModel, Field
from fastapi import FastAPI
import pandas as pd

sys.path.append(os.path.abspath(".."))
from utils.feature_engineer import FeatureEngineer

app = FastAPI()

with open("model.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)

class House(BaseModel):
    MSSubClass: int
    MSZoning: str
    LotFrontage: float
    LotArea: int
    Street: str
    Alley: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str
    MasVnrArea: float
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str 
    BsmtFinType1: str
    BsmtFinSF1: int
    BsmtFinType2: str
    BsmtFinSF2: int
    BsmtUnfSF: int
    TotalBsmtSF: int
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    firstFlrSF: int = Field(..., alias="1stFlrSF")
    secondFlrSF: int = Field(..., alias="2ndFlrSF")
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int 
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str
    GarageType: str
    GarageYrBlt: int
    GarageFinish: str
    GarageCars: int
    GarageArea: int
    GarageQual: str
    GarageCond: str
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    thirdSsnPorch: int = Field(..., alias="3SsnPorch")
    ScreenPorch: int
    PoolArea: int
    PoolQC: str
    Fence: str
    MiscFeature: str 
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str 

@app.post("/predict")
def predict(house: House):
    json = pd.DataFrame([house.dict(by_alias=True)])
    prediction = pipeline.predict(json)[0]
    return f"House price = {float(prediction)}"
