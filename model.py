# model.py
import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('Car details v3.csv')
df.dropna(inplace=True)

# Keep year and name
# Clean numeric columns
def clean_numeric(val):
    if isinstance(val, str):
        val = re.findall(r"[\d.]+", val)
        return float(val[0]) if val else np.nan
    return val

for col in ['mileage', 'engine', 'max_power']:
    df[col] = df[col].apply(clean_numeric)
    df[col].fillna(df[col].median(), inplace=True)

df.drop(['torque'], axis=1, inplace=True)

# Encode name
le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])

# Encode owner
df['owner'] = df['owner'].map({
    'Test Drive Car': 0,
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4
})

# One-hot encode
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission'], drop_first=True)

# Split
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)

# Choose best model
models = {'Linear Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}
scores = {k: r2_score(y_test, m.predict(X_test)) for k, m in models.items()}
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

print(f"✅ Best model: {best_model_name}")

# Save files
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

with open('label_encoder_name.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Model, scaler, label encoder, and columns saved.")
