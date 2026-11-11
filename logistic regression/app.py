# ==========================================
# DIABETES PREDICTION STREAMLIT APPLICATION
# ==========================================

import pickle
import pandas as pd
import streamlit as st
from pathlib import Path

# ------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("Diabetes Prediction App")

st.write(
    "Enter the patient's measurements below to obtain "
    "a prediction from the trained Logistic Regression model."
)

st.info(
    "This application is for educational purposes only. "
    "It does not provide a medical diagnosis."
)

# ------------------------------------------
# 2. LOAD THE TRAINED MODEL
# ------------------------------------------

MODEL_PATH = Path(__file__).parent / "logistic_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error(
        "The file logistic_model.pkl was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

# ------------------------------------------
# 3. PATIENT INPUTS
# ------------------------------------------

st.subheader("Patient Information")

with st.form("diabetes_prediction_form"):

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=79.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

    submitted = st.form_submit_button("Predict Diabetes Outcome")

# ------------------------------------------
# 4. PREDICTION
# ------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    try:
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]
        diabetes_probability = probabilities[1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "The model predicts Outcome = 1 "
                "(positive diabetes classification)."
            )
        else:
            st.success(
                "The model predicts Outcome = 0 "
                "(negative diabetes classification)."
            )

        st.write(
            "Model-estimated probability for Outcome = 1:"
        )

        st.metric(
            "Estimated probability",
            f"{diabetes_probability:.2%}"
        )

        st.caption(
            "The probability is a model estimate, not a confirmed "
            "diagnosis or a substitute for clinical assessment."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# ------------------------------------------
# 5. FOOTER
# ------------------------------------------

st.divider()

st.caption(
    "Logistic Regression | Diabetes Outcome Prediction | "
    "Educational Machine Learning Project"
)