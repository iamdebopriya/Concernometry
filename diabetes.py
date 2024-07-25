import streamlit as st
import pickle
import numpy as np

# Load the model
with open('svm_diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)



# Streamlit app
def main():
    # Add inline CSS for background gradient
    st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(to right, #ffc0cb, #dda0dd); /* Light pink to light purple gradient */
            }
            .stTitle {
                color: #800080; /* Title color */
                font-size: 2em; /* Title size */
            }
            .stButton > button {
                background-color: #800080; /* Button background color */
                color: #fff; /* Button text color */
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
            }
            .stButton > button:hover {
                background-color: #4b0082; /* Button hover color */
            }
            </style>
        """, unsafe_allow_html=True)
    st.title("Concernometry:-Show Concern About Your Diabetes")
    st.write("""
        ## About this App
        Welcome to the Innovative Female Diabetes Predictor! This application helps you assess the likelihood of having diabetes based on various health parameters specifically for women.

        **How to use the app:**
        1. Enter your details such as the number of pregnancies, glucose level, blood pressure, skin thickness, BMI, diabetes pedigree function, age, and insulin level.
        2. Click on the "Predict" button.
        3. The app will display the probability of being diabetic and non-diabetic, along with an overall prediction of your health status.

        Please note that this prediction is based on statistical models and should not replace professional medical advice. Always consult with a healthcare provider for medical concerns.
        """)

    st.write("Enter the following details to predict diabetes:")

    # Input fields
    pregnancy = st.number_input("Pregnancy(0-20)", min_value=0, max_value=20)
    glucose = st.number_input("Glucose(0-200)", min_value=0, max_value=200)
    blood_pressure = st.number_input("Blood Pressure(0-200)", min_value=0, max_value=200)
    skin_thickness = st.number_input("Skin Thickness(0-100)", min_value=0, max_value=100)
    bmi = st.number_input("BMI(0-50)", min_value=0.0, max_value=50.0, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function(0-2.5)", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120)
    insulin = st.number_input("Insulin(0-1000)", min_value=0, max_value=1000)

    # Predict button
    if st.button("Predict"):
        features = np.array(
            [pregnancy, glucose, blood_pressure, skin_thickness,insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
        proba = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        st.header(f"Probability of Diabetic: {proba[1]:.2f}")
        st.header(f"Probability of Non-Diabetic: {proba[0]:.2f}")
        if prediction == 1:
            st.write("Overall Statistics: The person is predicted to be **Diabetic**.")
            st.write("""
                    ### Advice for Managing Diabetes
                    - **Diet**: Maintain a balanced diet rich in vegetables, whole grains, and lean proteins. Avoid sugary foods and beverages.
                    - **Exercise**: Engage in regular physical activity such as walking, jogging, or yoga for at least 30 minutes a day.
                    - **Weight Management**: Aim to achieve and maintain a healthy weight through a combination of diet and exercise.
                    - **Monitoring**: Regularly monitor your blood sugar levels to keep track of your health.
                    - **Medication**: Follow your healthcare provider's advice regarding any medications or insulin therapy.
                    - **Hydration**: Drink plenty of water and avoid sugary drinks.
                    - **Sleep**: Ensure you get enough sleep, as poor sleep can affect blood sugar levels.
                    - **Stress Management**: Practice stress-reducing activities such as meditation, deep breathing, or hobbies you enjoy.
                    """)
        else:
            st.write("Overall Statistics: The person is predicted to be **Non-Diabetic**.")
        st.write("Thank you for using Concernometry. Be well and take care!")


if __name__ == "__main__":
    main()
