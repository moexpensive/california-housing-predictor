# 🏡 California Housing Price Predictor

An end-to-end Machine Learning project built using **Scikit-Learn**, **Flask**, and **Joblib**, based on the *Hands-On Machine Learning with Scikit-Learn & TensorFlow* book.

## 🔍 Overview
This project predicts California housing prices using 1990 census data.  
Includes:
- Custom Transformers (`ClusterSimilarity`, `column_ratio`)
- Full preprocessing pipeline
- Random Forest regression
- Flask app for web-based predictions

## ⚙️ Tech Stack
- Python (Scikit-Learn, Flask, Joblib, Pandas, NumPy)
- HTML (Jinja2 templates)
- Render (for hosting)
- GitHub (for version control)

## 💻 How to Run Locally
```bash
pip install -r requirements.txt
python app.py

Go to http://127.0.0.1:5000

📦 Files
File	Description
app.py	Flask web app
model/my_california_housing_model.pkl	Serialized model
housing.ipynb	Notebook used for data analysis and model training
templates/index.html	User interface for predictions
🌐 Live Demo

(Will add once deployed)

Project inspired by Aurélien Géron’s "Hands-On Machine Learning".

