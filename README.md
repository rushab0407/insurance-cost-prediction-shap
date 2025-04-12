# 🏥 Medical Insurance Cost Prediction with SHAP

This is an interactive **Streamlit web app** that predicts an individual's medical insurance cost using a **Random Forest Regressor**, and explains the prediction using **SHAP (SHapley Additive Explanations)**


![Insurance Dashboard]asset/shap.png
---

##  Features

- 🔢 Predicts insurance costs based on:
  - Age
  - Sex
  - BMI
  - Number of children
  - Smoking status
  - Region
- ⚙️ Powered by a trained Random Forest Regressor (sklearn)
- 💡 Provides interactive SHAP waterfall plot to explain individual predictions
- 📊 Built with Streamlit for a smooth UI experience

---

- Python
- Pandas, NumPy
- Scikit-learn
- SHAP
- Streamlit
- Matplotlib
- Joblib

---

##  Dataset Info

- **Source**: [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Contains:
  - Demographics (`age`, `sex`, `region`)
  - Health data (`bmi`, `children`, `smoker`)
  - Target variable: `charges` (insurance cost)

---

##  Model Performance

- ✅ **R² Score**: ~0.87
- ✅ **MAE**: ~$2550
- Model: `RandomForestRegressor(n_estimators=100)`

---

