import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv("P:\Machine Learning\house_prices_dataset.csv")


print("Columns : \n", data.head())
print("\n\nSummary of Dataset : \n", data.describe())
print("\n\nInfo : ", data.info())
print("\n\nMissing Values : \n", data.isnull().sum())
print("\n\nShape : \n", data.shape)
print("\n\nColumns : \n",data.columns)


x = data[['SquareFootage']]

y = data[['Price']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print(f"\n\nMean Squared Error : {mse}")

plt.scatter(x_test,y_test, color = 'blue', label = 'Test Data')
plt.plot(x_test,y_pred, color = 'red', label = 'Regression Line')
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("House Price Prediction Model using Linear Regression")
plt.show()