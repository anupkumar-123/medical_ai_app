import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#lOAD dATASET

diabetes = pd.read_csv("diabetes.csv")
heart = pd.read_csv("heart.csv")

diabetes.head()
print(diabetes.head())
print(heart.head())


def tarin_model(data, filename):
    x = data.drop("target", axis=1)
    y = data["target"]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    joblib.dump(model, f"{filename}")
    print("Model traines successfully")

tarin_model(diabetes, "diabetes.pkl")
tarin_model(heart, "heart.pkl")

    