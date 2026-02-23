from flask import Flask, render_template, request
import pickle
import numpy as np



app = Flask(__name__)

# Load model only (remove scaler)
model = pickle.load(open("Rainfall.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        MinTemp = float(request.form["MinTemp"])
        MaxTemp = float(request.form["MaxTemp"])
        Rainfall = float(request.form["Rainfall"])

        features = np.array([[MinTemp, MaxTemp, Rainfall]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            return render_template("chance.html")
        else:
            return render_template("noChance.html")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)