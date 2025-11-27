
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi Harga Mobil Premium",
    page_icon="🚗",
    layout="wide"
)

# ======================
#  Load Model
# ======================
@st.cache_resource
def load_model():
    return joblib.load("model_car_price.pkl")

model = load_model()

# ======================
#  Header
# ======================
st.title("🚗 Prediksi Harga Mobil — Premium Edition")
st.write("Masukkan spesifikasi mobil untuk mendapatkan prediksi harga.")

# ======================
#  Sidebar Input
# ======================
st.sidebar.header("Input Spesifikasi Mobil")

year = st.sidebar.slider("Tahun Produksi", 1990, 2024, 2015)
mileage = st.sidebar.slider("Jarak Tempuh (km)", 0, 300000, 50000)
enginesize = st.sidebar.slider("Kapasitas Mesin (CC)", 600, 6000, 1500)
fueltype = st.sidebar.selectbox("Jenis Bahan Bakar", ["gas", "diesel"])
aspiration = st.sidebar.selectbox("Tipe Mesin", ["std", "turbo"])
brand = st.sidebar.selectbox(
    "Merek Mobil",
    ["toyota", "honda", "bmw", "audi", "mercedes", "hyundai", "nissan", "volkswagen"]
)

# ======================
#  Dataframe Input
# ======================
input_df = pd.DataFrame([{
    "year": year,
    "mileage": mileage,
    "enginesize": enginesize,
    "fueltype": fueltype,
    "aspiration": aspiration,
    "brand": brand
}])

st.subheader("📋 Spesifikasi Mobil")
st.table(input_df)

# ======================
#  Prediction
# ======================
if st.sidebar.button("🔮 Prediksi Harga Mobil"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Harga Mobil Diprediksi: **Rp {prediction:,.2f}**")
