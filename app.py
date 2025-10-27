from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# Import your custom transformers
from custom_transformers import column_ratio, ratio_name, ClusterSimilarity

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/my_california_housing_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        input_data = [
            float(request.form["longitude"]),
            float(request.form["latitude"]),
            float(request.form["housing_median_age"]),
            float(request.form["total_rooms"]),
            float(request.form["total_bedrooms"]),
            float(request.form["population"]),
            float(request.form["households"]),
            float(request.form["median_income"]),
            request.form["ocean_proximity"]
        ]

        # Convert input to DataFrame (column names must match training data)
        columns = [
            "longitude", "latitude", "housing_median_age",
            "total_rooms", "total_bedrooms", "population",
            "households", "median_income", "ocean_proximity"
        ]

        final_input = pd.DataFrame([input_data], columns=columns)

        # Make prediction
        prediction = model.predict(final_input)
        output = round(prediction[0], 2)

        return render_template("result.html", prediction_text=f"Predicted house value: ${output:,}")

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
