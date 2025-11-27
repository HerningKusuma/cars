import streamlit as st
import pandas as pd
import pickle

# ================================
# Load the trained Random Forest model (using pickle)
# ================================
with open('random_forest_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ================================
# Streamlit UI
# ================================
st.title('Prediksi Harga Mobil (Car Price Prediction)')
st.write('Aplikasi untuk memprediksi harga mobil berdasarkan parameter yang diberikan.')

# Sidebar for user inputs
st.sidebar.header('Input Parameter Mobil')

def user_input_features():
    year = st.sidebar.slider('Tahun Produksi', 1990, 2024, 2015)
    mileage = st.sidebar.slider('Jarak Tempuh (km)', 0, 300000, 50000)
    engine_size = st.sidebar.slider('Kapasitas Mesin (L)', 0.5, 6.0, 1.5)

    fuel_type = st.sidebar.selectbox('Tipe Bahan Bakar', ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
    transmission = st.sidebar.selectbox('Transmisi', ['Manual', 'Automatic'])
    brand = st.sidebar.selectbox('Merek Mobil', ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi', 'Hyundai'])

    data = {
        'year': year,
        'mileage': mileage,
        'engine_size': engine_size,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'brand': brand
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# ================================
# EXACT columns used in training (with OHE)
# ================================
training_columns_and_dtypes = {
    'year': 'int64',
    'mileage': 'int64',
    'engine_size': 'float64',

    # Fuel type OHE
    'fuel_type_Diesel': 'bool',
    'fuel_type_Electric': 'bool',
    'fuel_type_Hybrid': 'bool',
    'fuel_type_Petrol': 'bool',

    # Transmission OHE
    'transmission_Automatic': 'bool',
    'transmission_Manual': 'bool',

    # Brand OHE
    'brand_Audi': 'bool',
    'brand_BMW': 'bool',
    'brand_Honda': 'bool',
    'brand_Hyundai': 'bool',
    'brand_Mercedes': 'bool',
    'brand_Toyota': 'bool'
}

# ================================
# Create final input DataFrame
# ================================
final_input_df = pd.DataFrame(columns=training_columns_and_dtypes.keys())

# Assign dtype
for col, dtype in training_columns_and_dtypes.items():
    final_input_df[col] = final_input_df[col].astype(dtype)

# Default row
final_input_df.loc[0] = 0
for col, dtype in training_columns_and_dtypes.items():
    if dtype == 'bool':
        final_input_df.loc[0, col] = False

# Fill numerical inputs
final_input_df.loc[0, 'year'] = df_input['year'][0]
final_input_df.loc[0, 'mileage'] = df_input['mileage'][0]
final_input_df.loc[0, 'engine_size'] = df_input['engine_size'][0]

# Fill categorical inputs (One-Hot Encoding)
fuel_col = f"fuel_type_{df_input['fuel_type'][0]}"
if fuel_col in final_input_df.columns:
    final_input_df.loc[0, fuel_col] = True

trans_col = f"transmission_{df_input['transmission'][0]}"
if trans_col in final_input_df.columns:
    final_input_df.loc[0, trans_col] = True

brand_col = f"brand_{df_input['brand'][0]}"
if brand_col in final_input_df.columns:
    final_input_df.loc[0, brand_col] = True

# ================================
# Prediction
# ================================
if st.sidebar.button('Prediksi Harga Mobil'):
    try:
        prediction = model.predict(final_input_df)
        st.subheader('Hasil Prediksi Harga Mobil:')
        st.write(f"Harga Mobil Diprediksi: Rp {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.exception(e)
