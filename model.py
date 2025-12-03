import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    df = pd.read_csv("data/salary.csv")

    X = df[['experience']]
    y = df['salary']

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "salary_model.pkl")
    print("Model trained and saved as salary_model.pkl")

if __name__ == "__main__":
    train_model()
