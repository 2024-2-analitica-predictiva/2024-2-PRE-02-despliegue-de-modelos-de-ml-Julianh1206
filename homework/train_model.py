# train_model.py

"""Build, deploy and access a model using scikit-learn"""

## libreria pickle para guardar el modelo
import pickle

import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

df = pd.read_csv("files/input/house_data.csv", sep=",")

## variables que vamos a usar para predecir el precio de la casa
features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

## variable que queremos predecir
target = df[["price"]]

## modelo de regresion lineal
estimator = LinearRegression()

## entrenamos el modelo donde x = predictoras y y = respuesta
estimator.fit(features, target)

with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(estimator, file)