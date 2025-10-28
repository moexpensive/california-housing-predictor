# California Housing Price Predictor 🏡

A machine learning web app built with **Scikit-Learn**, **Flask**, and **Joblib** that predicts median house values in California based on census data.

## Features
- End-to-end ML pipeline with preprocessing, feature engineering, and RandomForest regression.
- Custom transformers: `ClusterSimilarity`, ratio features, and log-scaled attributes.
- Flask web interface for easy input and prediction.

## Project Structure
housing_project/
├── app.py
├── model/
│ └── my_california_housing_model.pkl
├── templates/
│ └── index.html
├── requirements.txt
└── README.md