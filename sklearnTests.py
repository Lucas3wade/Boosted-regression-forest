from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = pd.read_csv('winequality-red.csv', skiprows=1, sep=';', header=None)
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values
rmse = []
mse = []
mae = []
accuracy = []
for i in range(10):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  regressor = RandomForestRegressor(n_estimators=100)
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)
  rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  mse.append(metrics.mean_squared_error(y_test, y_pred))
  mae.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  i=0
  results = zip(y_test,y_pred)
  for y1, y2 in results:
      #print(y1, ":", y2)
      if round(y2) == y1:
        i+=1
  #print("accuracy: ",i/len(y_test), i, len(y_test))
  accuracy.append(i/len(y_test)*100)
  
print('Red wines with sklearn test:trening 3:7')   
print('Mean Mean Absolute Error:', sum(mae)/len(mae))
print('Mean Mean Squared Error:', sum(mse)/len(mse))
print('Mean Root Mean Squared Error:', sum(rmse)/len(mae))
print('Mean Accuracy:', sum(accuracy)/len(accuracy))

dataset = pd.read_csv('winequality-white.csv', skiprows=1, sep=';', header=None)
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values
rmse = []
mse = []
mae = []
accuracy = []
for i in range(10):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  regressor = RandomForestRegressor(n_estimators=100)
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)
  rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  mse.append(metrics.mean_squared_error(y_test, y_pred))
  mae.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

  i=0
  results = zip(y_test,y_pred)
  for y1, y2 in results:
      if round(y2) == y1:
        i+=1
  accuracy.append(i/len(y_test)*100)
print('White wines with sklearn test:trening 3:7') 
print('Mean Mean Absolute Error:', sum(mae)/len(mae))
print('Mean Mean Squared Error:', sum(mse)/len(mse))
print('Mean Root Mean Squared Error:', sum(rmse)/len(mae))
print('Mean Accuracy:', sum(accuracy)/len(accuracy))


