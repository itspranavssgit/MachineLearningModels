import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

import opendatasets as od
df= od.download('https://www.kaggle.com/datasets/altavish/boston-housing-dataset')

import pandas as pd
df=pd.read_csv('/content/boston-housing-dataset/HousingData.csv')
print(f"\nMissing values:\n{df.isnull().sum()}")
df.fillna(df.mean(), inplace=True)
sns.boxplot(x=df['RM'])
print("\nBasic statistics:")
print(df.describe())
target_column='medv' if 'medv' in df.columns else df.columns[-1]
X=df.drop(target_column, axis=1)
y=df[target_column]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(f"Features:{list(X.columns)}")
print(f"Target:{target_column}")
print(f"Target:{target_column}")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(f"Training samples:{len(X_train)}")
print(f"Testing samples:{len(X_test)}")
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
def predict_by_rooms(rm_value):
  """Predict price using only RM value (other features = median)"""
  features=X.median().copy()
  features['RM']=rm_value
  return model.predict([features])[0]
print(f"\nTesting the Function:")
test_price=predict_by_rooms(7.5)
print(f"Predicted price for RM=7.5: ${test_price:.2f}")
