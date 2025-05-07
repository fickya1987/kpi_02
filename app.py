import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("kpi_cleaned.csv")
    df = df.dropna(subset=["TARGET TW TERKAIT", "REALISASI TW TERKAIT"])
    df["TARGET TW TERKAIT"] = pd.to_numeric(df["TARGET TW TERKAIT"], errors="coerce")
    df["REALISASI TW TERKAIT"] = pd.to_numeric(df["REALISASI TW TERKAIT"], errors="coerce")
    df = df.dropna(subset=["TARGET TW TERKAIT", "REALISASI TW TERKAIT"])
    return df

data = load_data()

# Encode categorical features
categorical_cols = ["POSISI PEKERJA", "PERUSAHAAN", "NAMA KPI", "POLARITAS"]
encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
for col, encoder in encoders.items():
    data[col] = encoder.transform(data[col])

# Feature selection
features = ["POSISI PEKERJA", "PERUSAHAAN", "NAMA KPI", "BOBOT", "TARGET TW TERKAIT", "POLARITAS"]
X = data[features]
y = data["REALISASI TW TERKAIT"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# UI
st.title("KPI Realization Prediction App")

# User inputs
user_input = {}
for col in ["POSISI PEKERJA", "PERUSAHAAN", "NAMA KPI", "POLARITAS"]:
    user_input[col] = st.selectbox(f"Select {col}", encoders[col].classes_.tolist())

user_input["BOBOT"] = st.slider("BOBOT", 0, 100, 20)
user_input["TARGET TW TERKAIT"] = st.number_input("TARGET TW TERKAIT", value=75.0)

# Encode inputs
input_data = pd.DataFrame([{
    col: encoders[col].transform([user_input[col]])[0] if col in encoders else user_input[col]
    for col in features
}])

# Prediction
prediction = model.predict(input_data)[0]
st.subheader("Predicted KPI Realization")
st.metric(label="REALISASI TW TERKAIT", value=round(prediction, 2))
