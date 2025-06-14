import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder

from utils.feature_engineer import FeatureEngineer

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

df = pd.read_csv("../data/train.csv")
df.info()

# Total entries 1460
(df.isna().sum() / df.shape[0]) * 100

# Columnms with nulls
# Lot Frontage, Alley,MasVnrType, MasVnrArea, BsmtQual, BsmtCond          
# BsmtExposure, BsmtFinType1, BsmtFinType2, Electrical, FireplaceQu      
# GarageType GarageYrBlt, GarageFinish, GarageQual, GarageCond        
# PoolQC, Fence, MiscFeature      

df.select_dtypes("number").describe()

# There are lot of outliers on different columns
df.select_dtypes("O").describe()
# Lots of Categoriacal Data

# Visualizations
df.columns

# SalePrice distribution. Skewed to the right, outliers existence
sns.histplot(data=df, x="SalePrice", kde=True)

# Distribution for all numerical columns
df.select_dtypes("number").hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

# MSSubclass is numerical?
# Skewed: LotFrontage, LotArea, TotalBasementSF, 1flSF, GrLvArea
# Curious YearBuilt, there's a cap on 1950 for YearRemodAdd, MasVnrArea

df.select_dtypes("O").columns
df.head()
sns.countplot(data=df, x="OverallQual")
sns.countplot(data=df, x="OverallCond")
sns.countplot(data=df, x="SaleCondition")

# What's the difference between ovaerallcondition and overallquality?
# Overall quality refers to materials and finishes
# Overall condition refers to the real condition of the house

sns.scatterplot(data=df, x="SalePrice", y="OverallCond")
sns.scatterplot(data=df, x="SalePrice", y="OverallQual")

# Seems that Sale price depends more on the materials and finishes than the condition of the house
sns.scatterplot(data=df, x="SalePrice", y="LotArea")

# Also, lot Are is not directly correlated with SalePrice
plt.figure(figsize=(25, 25))
sns.heatmap(data=df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)

# Lots of interesting correlations and it seems that they are positive correlations
# High correlation with SalePrice: LotFrontage, LotArea, OverallQual, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1
# TotalBasemnt, 1stFSF, 2ndflsf, GrLivArea, FullBath, HalfBath, TotRoomAbvGrnd, Fireplaces, GarageYearBuilt, GarageCars, GaragLivArea
# WoodDeckSF, OpenPorchSF

# Correlated cols with SalePrice
df.columns
cor_cols = ["LotFrontage", "LotArea", "OverallQual", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1",
            "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "FullBath", "HalfBath", "TotRmsAbvGrd", 
            "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF"]


for col in cor_cols:
    sns.scatterplot(data=df, x="SalePrice", y=col)
    plt.show()

sns.boxplot(data=df, x="OverallQual", y="SalePrice")

# Data cleaning
df_cleaned = df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"])

# Preprocessing
# Apply log LotFrontage, LotArea, TotalBasementSF, 1flSF, GrLvArea
cat_cols = df_cleaned.select_dtypes("O").columns
num_cols = df_cleaned.select_dtypes("number").columns
to_log_cols = ["LotFrontage", "LotArea", "TotalBasementSF", "Fence", "MiscFeature"]

log_transformer = Pipeline(steps=[
    ("log_transformer", FeatureEngineer())
])

cat_transformer = Pipeline(steps=[
    ("onehot_transformer", OneHotEncoder(handle_unknown="ignore")),
    ("imputer", SimpleImputer(strategy="constant", fill_value="NA"))
])

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])
