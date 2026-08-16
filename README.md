# Heart Disease Prediction App

A Streamlit web app that predicts a patient's risk of heart disease from clinical inputs, using a trained K-Nearest Neighbors (KNN) classification model.

## Overview

Takes standard clinical measurements (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, oldpeak, and ST slope) and returns a Low Risk / High Risk prediction in real time through an interactive UI.

## How It Works

1. User inputs are collected through sliders and dropdowns in the Streamlit UI
2. Categorical inputs are one-hot encoded to match the model's expected feature set (`columns.pkl`)
3. Missing columns are filled with zeros and features are reordered to match training-time column order
4. Inputs are scaled using the saved `StandardScaler` (`scaler.pkl`)
5. The trained KNN model (`KNN_heart.pkl`) predicts risk and the result is displayed with a success/error message

## Tech Stack

Python • Streamlit • Pandas • Scikit-learn (KNN, StandardScaler) • Joblib

## Files

| File | Purpose |
|---|---|
| `HeartDiseasePredictionApp.py` | Main Streamlit app |
| `KNN_heart.pkl` | Trained KNN classification model |
| `scaler.pkl` | Fitted StandardScaler used to scale numeric inputs |
| `columns.pkl` | Expected feature column order used at inference time |

## Running Locally

```bash
pip install streamlit pandas scikit-learn joblib
streamlit run HeartDiseasePredictionApp.py
```

The app will open in your browser at `http://localhost:8501`.

## Disclaimer

This app is a machine learning demo built for educational purposes and is not a substitute for professional medical advice or diagnosis.

## Author

Yashasvi Sardana
