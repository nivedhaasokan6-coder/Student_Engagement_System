from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib

app = Flask(__name__)

df = pd.read_csv("student_engagement_dataset.csv")

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
@app.route("/")
def home():
    return send_file("index.html")
@app.route("/preprocess")
def preprocess():

    missing = df.isnull().sum().sum()

    return render_template(
        "preprocess.html",
        missing=missing
    )

@app.route("/analysis")
def analysis():

    login_avg = round(df["Login_Frequency"].mean(), 2)
    quiz_avg = round(df["Quiz_Performance_Average"].mean(), 2)

    return render_template(
        "analysis.html",
        login_avg=login_avg,
        quiz_avg=quiz_avg
    )

@app.route("/predictpage")
def predictpage():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():

    login = float(request.form["login"])
    forum = float(request.form["forum"])
    assignment = float(request.form["assignment"])
    resource = float(request.form["resource"])
    quiz = float(request.form["quiz"])

    data = [[
        login,
        forum,
        assignment,
        resource,
        quiz
    ]]

    prediction = model.predict(data)

    result = le.inverse_transform(prediction)[0]

    return render_template(
        "result.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)
