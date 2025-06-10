import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None).
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

sns.pairplot(data=df)

# Distribution for all numerical columns
df.select_dtypes("number").hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

# MSSubclass is numerical?
# Skewed: LotFrontage, LotArea, TotalBasementSF, 1flSF, GrLvArea
# Curious YearBuilt, there's a cap on 1950 for YearRemodAdd, MasVnrArea