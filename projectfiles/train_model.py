import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_excel("dataset/WeatherAUS.csv.xlsx")

# Keep only required columns
data = data[["MinTemp", "MaxTemp", "Rainfall", "RainTomorrow"]]

# Remove rows where target is missing
data = data.dropna(subset=["RainTomorrow"])

# Convert target to 0 and 1
data["RainTomorrow"] = data["RainTomorrow"].map({"No": 0, "Yes": 1})

# Split features and target
X = data[["MinTemp", "MaxTemp", "Rainfall"]]
y = data["RainTomorrow"]

# Handle missing values in features
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save files
pickle.dump(model, open("Rainfall.pkl", "wb"))
pickle.dump(scaler, open("scale.pkl", "wb"))
pickle.dump(imputer, open("imputer.pkl", "wb"))

print("Model Saved Successfully")