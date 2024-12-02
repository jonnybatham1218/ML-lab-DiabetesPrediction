import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data = pd.read_csv(r"./diabetes_prediction_dataset.csv")
data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
data["smoking_history"] = data["smoking_history"].map({
    "never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6
})

y = data['diabetes']
x = data.drop("diabetes", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Updated CSS with new color scheme
st.markdown("""
    <style>
    h1 {
        color: #3b8a5d;  /* Dark green for the main title */
        text-align: center;
        font-size: 3rem;
    }
    h3 {
        color: #d9534f;  /* Red for high-risk result */
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .low-risk {
        color: #5cb85c;  /* Green for low-risk result */
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stForm {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stSelectbox, .stSlider {
        width: 100%;
    }
    .stButton {
        background-color: #5bc0de; /* Light blue for submit button */
        color: white;
        padding: 12px 24px;
        font-size: 16px;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton:hover {
        background-color: #31b0d5; /* Darker blue on hover */
    }
    .stColumns {
        display: flex;
        justify-content: center;
        gap: 40px;
    }
    .stColumn {
        width: 45%;
    }
    .stSlider, .stSelectbox {
        margin-bottom: 15px;
    }
    .stMarkdown {
        font-size: 1.2rem;
        text-align: center;
    }
    .feature-importance-bar {
        background-color: #3b8a5d; /* Dark green for feature importance bars */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Diabetes Prediction")

with st.form("my_form"):

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    spacer, col1, spacer, col2, spacer = st.columns([0.2, 1, 0.1, 1, 0.2])

    with col1:
        age = st.slider("Age", 0, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension", [0, 1])
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)

    with col2:
        smoking_history = st.selectbox("Smoking History", ["Never", "No Info", "Current", "Former", "Ever", "Not Current"])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        hba1c = st.slider("HbA1c Level", 4.0, 15.0, 5.5)
        blood_glucose = st.slider("Blood Glucose Level", 50, 250, 100)

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    gender_map = {"Male": 1, "Female": 2, "Other": 3}
    smoking_map = {"Never": 1, "No Info": 2, "Current": 3, "Former": 4, "Ever": 5, "Not Current": 6}
    input_data = np.array([[age, gender_map[gender], smoking_map[smoking_history], hypertension, heart_disease, bmi, hba1c, blood_glucose]])

    prediction = rf_model.predict(input_data)
    result = "High risk of diabetes." if prediction[0] == 1 else "Low risk of diabetes."
    
    if prediction[0] == 1:
        st.markdown(f"<h3 style='color: #d9534f;'>{result}</h3>", unsafe_allow_html=True) 
    else:
        st.markdown(f"<h3 class='low-risk'>{result}</h3>", unsafe_allow_html=True) 
        
st.subheader("Feature Importance")

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns

plt.figure(figsize=(8, 6))
plt.title("Feature Importance", fontsize=16)
plt.bar(range(x.shape[1]), importances[indices], align="center", color="#3b8a5d") 
plt.xticks(range(x.shape[1]), features[indices], rotation=45, ha='right')
plt.tight_layout()

st.pyplot(plt)
