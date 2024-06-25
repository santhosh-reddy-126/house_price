import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn
data = pd.read_csv("kc_house_data.csv")
data=data.drop(["id","date","yr_renovated","zipcode"],axis=1)
data["sqft_living15"]=data["sqft_living"]-data["sqft_living15"]
data["sqft_lot15"]=data["sqft_lot"]-data["sqft_lot15"]
y=data.iloc[:,0].values
x=data.iloc[:,1:].values
from scipy.stats import pearsonr
correlations = []
p_values = []
# Calculate Pearson correlation and p-value for each feature
for feature in x.T:
    corr, p_value = pearsonr(feature, y)
    correlations.append(corr)
    p_values.append(p_value)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import cross_val_score
model5 = RandomForestRegressor(n_estimators=10)
model5.fit(x_train,y_train)
accuracies = cross_val_score(estimator = model5,X = x_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print("Accuracy on test set: "+str(round(model5.score(x_test,y_test)*100,3))+"%")

joblib.dump(model5, 'regression_model_updated.pkl')
print(sklearn.__version__)
print(data.head())