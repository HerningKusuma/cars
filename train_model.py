
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# =====================================
# 1. LOAD DATASET
# =====================================
df = pd.read_csv("CarPrice_Assignment.csv")

# =====================================
# 2. FEATURE ENGINEERING
# =====================================
# Extract brand
df['brand'] = df['CarName'].apply(lambda x: x.split()[0].lower())

# Create synthetic year (dataset has no year info)
df["year"] = 2024 - df["symboling"] * 5

# Create synthetic mileage using citympg
df["mileage"] = df["citympg"] * 1000

# =====================================
# 3. COLUMNS USED (MUST MATCH app.py)
# =====================================
X = df[['year', 'mileage', 'enginesize', 'fueltype', 'aspiration', 'brand']]
y = df['price']

numeric_features = ['year', 'mileage', 'enginesize']
categorical_features = ['fueltype', 'aspiration', 'brand']

# =====================================
# 4. PIPELINE (PREPROCESSING + MODEL)
# =====================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=250, random_state=42))
])

# =====================================
# 5. TRAIN MODEL
# =====================================
model.fit(X, y)

# =====================================
# 6. SAVE MODEL
# =====================================
joblib.dump(model, "model_car_price.pkl")

print("Model berhasil dilatih dan disimpan sebagai model_car_price.pkl!")
