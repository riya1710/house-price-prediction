import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('models/house_price_model.pkl')

# Load one sample row to get all columns the model expects
sample_data = pd.read_csv('data/train.csv').drop(['Id', 'SalePrice'], axis=1)
default_input = sample_data.iloc[[0]]  # one-row DataFrame

# Streamlit App
st.title("🏠 House Price Predictor")
st.markdown("Adjust the values below:")

# Update specific fields
default_input['GrLivArea'] = st.number_input('Above Grade Living Area (sq ft)', 500, 4000, 1500)
default_input['GarageCars'] = st.slider('Garage Capacity (cars)', 0, 4, 2)
default_input['TotalBsmtSF'] = st.number_input('Total Basement Area (sq ft)', 0, 3000, 800)
default_input['FullBath'] = st.slider('Number of Full Bathrooms', 0, 4, 2)
default_input['OverallQual'] = st.slider('Overall Quality (1-10)', 1, 10, 5)
default_input['YearBuilt'] = st.slider('Year Built', 1900, 2025, 1990)

# Show the edited row
st.subheader("📋 Final Input to Model")
st.write(default_input)

# Predict
if st.button("Predict Sale Price"):
    prediction = model.predict(default_input)
    st.subheader("💰 Predicted House Price:")
    st.success(f"${int(prediction[0]):,}")
