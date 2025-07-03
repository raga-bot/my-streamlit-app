import streamlit as st
import numpy as np
import joblib
import pickle # Import pickle to load the scaler

# Function to load the model and scaler
# Using st.cache_resource to cache the loaded model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = None
    scaler = None
    try:
        model = joblib.load("diabetes_model_rf.pkl")
    except FileNotFoundError:
        st.error("Error: 'diabetes_model_rf.pkl' not found. Please ensure the model file is in the same directory.")
        return None, None # Return None for both if model is not found

    try:
        # Load the scaler object that was fitted on the training data
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("Error: 'scaler.pkl' not found. Please ensure the scaler file is in the same directory.")
        return model, None # Return loaded model but None for scaler if scaler not found
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return model, None # Return loaded model but None for scaler if other error occurs

    return model, scaler

# Load the model and scaler
model, scaler = load_model_and_scaler()

# Proceed only if both model and scaler were loaded successfully
if model is not None and scaler is not None:
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter the patient's data below:")

    # Input fields
    # Ensure the min/max values and step are appropriate for your data
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    Glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=100.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=150.0, value=70.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
    Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    Age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    # Create a numpy array from the input data
    # The order of features must match the order used during training
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                            DiabetesPedigreeFunction, Age]])

    if st.button("Predict"):
        try:
            # Scale the input data using the loaded scaler
            # The scaler expects input in the same format it was trained on (e.g., a 2D array)
            input_data_scaled = scaler.transform(input_data)

            # Make prediction using the scaled input data
            prediction = model.predict(input_data_scaled)

            if prediction[0] == 1:
                st.error("❌ The model predicts the patient is likely to have diabetes.")
            else:
                st.success("✅ The model predicts the patient is unlikely to have diabetes.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Optional: Add a message if model or scaler could not be loaded
elif model is None or scaler is None:
    st.warning("Please ensure that 'diabetes_model_rf.pkl' and 'scaler.pkl' files are in the same directory as 'app.py'.")