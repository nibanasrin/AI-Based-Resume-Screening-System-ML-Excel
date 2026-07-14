import streamlit as st
import pickle
import numpy as np

# Load model
with open("resume_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("AI-Based Resume Screening System")

# User inputs
skills = st.number_input("Skills", min_value=0, max_value=300, value=100)
experience = st.number_input("Experience (Years)", min_value=0, max_value=20, value=5)
education = st.number_input("Education", min_value=0, max_value=5, value=2)
certifications = st.number_input("Certifications", min_value=0, max_value=10, value=1)
job_role = st.number_input("Job Role", min_value=0, max_value=10, value=1)
salary = st.number_input("Salary Expectation ($)", min_value=0, max_value=200000, value=50000)
projects = st.number_input("Projects Count", min_value=0, max_value=20, value=2)
ai_score = st.number_input("AI Score (0-100)", min_value=0, max_value=100, value=50)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[skills, experience, education, certifications,
                            job_role, salary, projects, ai_score]])

    prediction = model.predict(input_data)
    st.write("Prediction Value:", prediction[0])

    if prediction[0] == 0:
        st.success("Selected")
    else:
        st.error("Rejected")