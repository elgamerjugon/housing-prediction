import pandas as pd
import sys
import os
import cloudpickle

sys.path.append(os.path.abspath(".."))
from utils.feature_engineer import FeatureEngineer

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error

def generate_model():
    df= pd.read_csv("../data/train.csv")
    X = df.drop(columns=["Id", "Alley", "PoolQC", "Fence", "MiscFeature", "SalePrice"])
    y = df.SalePrice

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # df_cleaned = df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature", "SalePrice"])

    # Preprocessing
    # Apply log LotFrontage, LotArea, TotalBasementSF, 1flSF, GrLvArea
    cat_cols = X_train.select_dtypes("O").columns
    to_log_cols = ["LotFrontage", "LotArea", "TotalBsmtSF"]
    num_cols = X_train.select_dtypes("number").columns.difference(to_log_cols)

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot_transformer", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat_transformer", cat_transformer, cat_cols),
        ("num_transformer", num_transformer, num_cols),
        ("log_transformer", FeatureEngineer(columns=to_log_cols), to_log_cols)
    ], remainder="passthrough")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",RandomForestRegressor(max_depth=40, max_features='sqrt',
                                       min_samples_split=5))
    ])

    pipeline.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        cloudpickle.dump(pipeline, f)
    
    return pipeline

if __name__ == "__init__":
    pipeline = generate_model()
