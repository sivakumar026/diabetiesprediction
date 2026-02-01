import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
with open("D:/diabetiespredict/predict/model.sav", "rb") as f:
    load_model = pickle.load(f)

with open("D:/diabetiespredict/predict/scaler.sav", "rb") as f:
    scaler = pickle.load(f)


def diabeties_predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    std_data = scaler.transform(input_data_reshaped)
    prediction = load_model.predict(std_data)

    if prediction[0] == 0:
        return "The person is NOT diabetic"
    else:
        return "The person IS diabetic"


def main():
    st.title("Diabetes Prediction App")

    # User inputs
    pregnancies = st.text_input("Number of pregnancies")
    glucose = st.text_input("Glucose level")
    bloodpressure = st.text_input("Blood pressure")
    skinthickness = st.text_input("Skin thickness")
    insulin = st.text_input("Insulin level")
    bmi = st.text_input("BMI")
    diabetes_pedigree = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")

    diagnosis = ""

    if st.button("Diabetes Test Result"):
        diagnosis = diabeties_predict([
            pregnancies,
            glucose,
            bloodpressure,
            skinthickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ])

    st.success(diagnosis)


if __name__ == "__main__":
    main()
