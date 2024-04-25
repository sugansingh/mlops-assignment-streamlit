import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def Selling_Price(Car_Name,Year,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission):
    features = np.array([Car_Name,Year,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission])
    features = features.reshape(1,-1)
    Selling_Price = model.predict(features)
    return Selling_Price[0]

# Streamlit UI
st.title('Selling_Price')
st.write("""
## Input Features
Enter the values for the input features to predict EMISSION.
""")

# Input fields for user
Car_Name = st.number_input('Car_Name')
Year = st.number_input('year')
Present_Price = st.number_input('Present_Price')
Kms_Driven= st.number_input('kms_Driven')
Fuel_Type = st.number_input('Fuel_Type ')
Seller_Type = st.number_input('Seller_Typ')
Transmission = st.number_input('Transmission')

# Prediction button
if st.button('Predict'):
    # Predict Selling_Price
    Selling_Price_prediction =Selling_Price(Car_Name,Year,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission)
    st.write(f"Predicted Selling_Price: {Selling_Price_prediction}")