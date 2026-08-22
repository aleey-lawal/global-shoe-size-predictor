# 👟 Global Shoe Size Predictor

A multi-output machine learning app that predicts shoe sizes in EU, US, UK,
JP, and CN standards — trained on data grounded in Volumental's 1.2M foot
scan research study.

## 🔗 Live Demo
👉 [Open the App](https://aleey-lawal-global-shoe-predictor.streamlit.app)

## 🌍 Dataset
1,000 records across 5 global regions and 22 countries.
Foot length means calibrated to Jurca et al. (2019), Scientific Reports.

## 🤖 Model
- Type: Multi-Output Random Forest Regression
- Targets: EU, US, UK, JP (Mondopoint), CN sizes
- Average R²: 0.9958

## 🛠️ Tech Stack
Python · scikit-learn · Streamlit · Pandas · Plotly · Joblib

## 📁 Project Structure
- data/      → dataset (1,000 records across 22 countries)
- notebooks/ → EDA and model training notebook
- model/     → saved Random Forest pipeline
- app.py     → Streamlit web application