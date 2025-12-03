from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("salary_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    exp = float(request.form['experience'])
    prediction = model.predict([[exp]])
    predicted_salary = round(prediction[0], 2)

    return render_template("result.html",
                           experience=exp,
                           salary=predicted_salary)

if __name__ == '__main__':
    app.run(debug=True)
