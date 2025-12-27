from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

# EXACT mapping used during training
YES_NO = {
    "yes": 1,
    "no": 0
}

FURNISHING = {
    "furnished": 0,
    "semi-furnished": 1,
    "unfurnished": 2
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    area = int(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    stories = int(request.form["stories"])
    parking = int(request.form["parking"])

    mainroad = YES_NO[request.form["mainroad"].lower()]
    guestroom = YES_NO[request.form["guestroom"].lower()]
    basement = YES_NO[request.form["basement"].lower()]
    hotwaterheating = YES_NO[request.form["hotwaterheating"].lower()]
    airconditioning = YES_NO[request.form["airconditioning"].lower()]
    prefarea = YES_NO[request.form["prefarea"].lower()]
    furnishingstatus = FURNISHING[request.form["furnishingstatus"].lower()]

    input_data = np.array([[
        area, bedrooms, bathrooms, stories,
        mainroad, guestroom, basement,
        hotwaterheating, airconditioning,
        parking, prefarea, furnishingstatus
    ]])

    prediction = model.predict(input_data)[0]

    return f"Predicted House Price: {prediction:.2f}"

if __name__ == "__main__":
    app.run(debug=True)
