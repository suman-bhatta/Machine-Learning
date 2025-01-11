import pandas
from sklearn import linear_model

# Load the dataset
df = pandas.read_csv("data.csv")

# Define independent variables (X) and dependent variable (y)
X = df[['Weight', 'Volume']]
y = df['CO2']

# Create a linear regression model and fit it to the data
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Predict the CO2 emission of a car with specified weight and volume
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
