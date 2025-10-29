import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("model/my_california_housing_model.pkl")

# Prediction function
def predict(median_income, housing_median_age, total_rooms, total_bedrooms, population, households, latitude, longitude):
    features = np.array([[median_income, housing_median_age, total_rooms, total_bedrooms,
                          population, households, latitude, longitude]])
    prediction = model.predict(features)
    return f"Predicted Median House Value: ${prediction[0]:,.2f}"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Median Income"),
        gr.Number(label="Housing Median Age"),
        gr.Number(label="Total Rooms"),
        gr.Number(label="Total Bedrooms"),
        gr.Number(label="Population"),
        gr.Number(label="Households"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude")
    ],
    outputs="text",
    title="California Housing Price Predictor",
    description="Enter housing data and predict the median house value using a trained model."
)

iface.launch()
