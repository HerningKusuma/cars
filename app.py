import streamlit as st
import pandas as pd
import pickle

# Load trained Random Forest Regression model
with open("random_forest_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🚗 Prediksi Harga Mobil")
st.write("Masukkan spesifikasi mobil untuk memprediksi harganya.")

# Sidebar Input
st.sidebar.header("Input Fitur Mobil")

def user_input():
    wheelbase = st.sidebar.slider("Wheelbase (mm)", 80.0, 130.0, 100.0)
    carlength = st.sidebar.slider("Car Length (mm)", 120.0, 200.0, 150.0)
    carwidth = st.sidebar.slider("Car Width (mm)", 50.0, 90.0, 65.0)
    curbweight = st.sidebar.slider("Curb Weight (kg)", 1000, 5000, 2000)
    enginesize = st.sidebar.slider("Engine Size (cc)", 50, 500, 150)
    boreratio = st.sidebar.slider("Bore Ratio", 2.0, 5.0, 3.0)
    horsepower = st.sidebar.slider("Horsepower", 40, 300, 100)
    citympg = st.sidebar.slider("City MPG", 5, 60, 25)
    highwaympg = st.sidebar.slider("Highway MPG", 5, 70, 30)

    data = {
        "wheelbase": wheelbase,
        "carlength": carlength,
        "carwidth": carwidth,
        "curbweight": curbweight,
        "enginesize": enginesize,
        "boreratio": boreratio,
        "horsepower": horsepower,
        "citympg": citympg,
        "highwaympg": highwaympg
    }

    df = pd.DataFrame(data, index=[0])
    return df

df_input = user_input()

st.subheader("📌 Input Anda:")
st.write(df_input)

# Predict
if st.sidebar.button("Prediksi Harga Mobil"):
    prediction = model.predict(df_input)
    st.subheader("💰 Prediksi Harga Mobil:")
    st.success(f"Rp {prediction[0]:,.2f}")
