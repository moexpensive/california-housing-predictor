import joblib
model = joblib.load("model/my_california_housing_model.pkl")
joblib.dump(model, "model/my_california_housing_model_compressed.pkl", compress=3)
