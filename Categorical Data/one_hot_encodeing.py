import pandas as pd 

cars = pd.read_csv('data.csv')
ohe_cars = pd.get_dummies(cars, columns = ['Make', 'Colour', 'Doors'])

print(ohe_cars.to_string())