import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

x = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(x, y)

print(regr.coef_)


"Copy the example from before, but change the weight from 2300 to 3300:"

import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")
x = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(x, y)

predicatedCO2 = regr.predict([[3300, 1300]])

print(predicatedCO2)