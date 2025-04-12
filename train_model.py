import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import shap

df = pd.read_csv("/Users/rushabarram/Documents/insurance.csv")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, "shap_explainer.pkl")

print("Model and SHAP explainer saved!")
