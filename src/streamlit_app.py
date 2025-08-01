import streamlit as st
import pandas as pd
import mlflow.sklearn

st.title("Breast Cancer Diagnosis Prediction")

# Replace <RUN_ID> with your actual run id from MLflow training output
model = mlflow.sklearn.load_model("runs:/4838809c53f448c69a08c3b732b9a5a6/rf-model")

features = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
    'concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
    'perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst',
    'concave points_worst','symmetry_worst','fractal_dimension_worst'
]
input_data = {}
for feat in features:
    input_data[feat] = st.number_input(feat, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.write("Prediction:", "Malignant" if prediction == 1 else "Benign")
