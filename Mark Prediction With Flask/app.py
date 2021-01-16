from flask import Flask, render_template, redirect, request
import joblib
import math

#__name__ = __main__
app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/", methods = ['POST'])
def marks():
    if request.method == 'POST':
        hours = float(request.form['hours'])
        prediction = math.floor(model.predict([[hours]])[0][0])
        marks = str(prediction)

    return render_template("index.html", your_marks = marks)

if __name__ == '__main__':
    app.run(debug = True)