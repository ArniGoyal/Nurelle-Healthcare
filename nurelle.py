import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import base64
from io import BytesIO

# Load datasets
medicine_df = pd.read_csv('medicine.csv')
diet_df = pd.read_csv('diet.csv')
yoga_df = pd.read_csv('yoga.csv')

# Load logo
logo = Image.open("nurelle_logo.png")
st.set_page_config(page_title="Nurelle: Healthcare Chatbot", layout="centered")

def logo_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# -----------------------
# 💫 Styling
# -----------------------
st.markdown("""
    <style>
    .main { background-color: #fef6f9; }
    h1, h2, h3, h4 { color: #4a4e69; }
    .stButton button { background-color: #ff8fab; color: white; border-radius: 10px; }
    .stTextInput, .stSelectbox { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# 🔮 Disease Prediction Model Setup
# -----------------------

medicine_df = medicine_df[medicine_df['Outcome Variable'] != 'Negative']
medicine_df = medicine_df.drop('Outcome Variable', axis=1)

# Balance rare diseases
disease_counts = medicine_df['Disease'].value_counts()
rare = disease_counts[disease_counts == 1].index.tolist()
medicine_df = pd.concat([medicine_df, medicine_df[medicine_df['Disease'].isin(rare)]], ignore_index=True)

cat_feat_m = ['Fever','Cough','Fatigue','Difficulty Breathing','Gender','Blood Pressure', 'Cholesterol Level']
num_feat_m = ['Age']

preprocessor_m = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat_m),
    ('num', 'passthrough', num_feat_m)
])

pipeline_m = Pipeline([
    ('preprocessor', preprocessor_m),
    ('classifier', RandomForestClassifier(max_samples=0.75, random_state=42))
])

X_m = medicine_df.drop('Disease', axis=1)
y_m = medicine_df['Disease']
pipeline_m.fit(X_m, y_m)

# -----------------------
# 🥗 Diet Recommendation Model
# -----------------------
diet_df = diet_df.drop(['Patient_ID'], axis=1)
diet_df = diet_df.dropna(subset=['Dietary_Restrictions', 'Disease_Type'])
diet_df['Allergies'] = diet_df['Allergies'].fillna("No")

cat_feat_d = ['Gender','Disease_Type','Severity','Physical_Activity_Level','Dietary_Restrictions','Allergies','Preferred_Cuisine']
num_feat_d = ['Age','Weight_kg','Height_cm','BMI','Daily_Caloric_Intake','Cholesterol_mg/dL',
              'Blood_Pressure_mmHg','Glucose_mg/dL','Weekly_Exercise_Hours','Adherence_to_Diet_Plan','Dietary_Nutrient_Imbalance_Score']

preprocessor_d = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat_d),
    ('num', 'passthrough', num_feat_d)
])

pipeline_d = Pipeline([
    ('preprocessor', preprocessor_d),
    ('classifier', RandomForestClassifier(max_samples=0.75, random_state=42))
])

X_d = diet_df.drop('Diet_Recommendation', axis=1)
y_d = diet_df['Diet_Recommendation']
pipeline_d.fit(X_d, y_d)

# -----------------------
# 🧘 Yoga Mapping
# -----------------------
yoga_map = yoga_df.groupby('Disease/Condition').apply(
    lambda x: x[['Recommended Yoga Practice', 'Frequency', 'Estimated Duration to Benefit', 'Observed Benefit']].to_dict('records')
).to_dict()

# -----------------------
# 💻 App UI
# -----------------------

st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{logo_to_base64(logo)}' width='500'/>
        <h1 style='color:#858ec9;'>Nurelle: Your Healthcare Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("Welcome to **Nurelle**, your personal AI healthcare buddy. I’ll help you predict diseases, recommend diet plans, suggest yoga, and even prescribe suitable medicines 💊🧘‍♀️.")

st.header("🩺 Health Check - Enter your symptoms")

with st.form("health_form"):
    fever = st.selectbox("Fever", ["Yes", "No"])
    cough = st.selectbox("Cough", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    breath = st.selectbox("Difficulty Breathing", ["Yes", "No"])
    age = st.slider("Age", 0, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.selectbox("Blood Pressure", ["Normal", "High", "Low"])
    chol = st.selectbox("Cholesterol Level", ["Normal", "High","Low"])
    med_type = st.selectbox("Preferred Medicine Type", ["Allopathic", "Ayurvedic", "Homeopathic"])

    submitted = st.form_submit_button("Get Health Insights")

if submitted:
    user_input = pd.DataFrame([{
        'Fever': fever,
        'Cough': cough,
        'Fatigue': fatigue,
        'Difficulty Breathing': breath,
        'Age': age,
        'Gender': gender,
        'Blood Pressure': bp,
        'Cholesterol Level': chol
    }])

    disease = pipeline_m.predict(user_input)[0]
    st.success(f"🧬 Predicted Disease: **{disease}**")

    # Medicine
    medicine_data = pd.read_csv('rec_medicines.csv')
    med_row = medicine_data[medicine_data['Disease'] == disease]
    if not med_row.empty:
        st.info(f"💊 Recommended {med_type} medicine: {med_row[med_type].values[0]}")
    else:
        st.warning("No medicine data found for this disease.")

    # Diet
    diet_sample = diet_df[diet_df['Disease_Type'] == disease]
    if not diet_sample.empty:
        diet_sample = diet_sample.sample(1, random_state=1)
        predicted_diet = pipeline_d.predict(diet_sample.drop('Diet_Recommendation', axis=1))[0]
        st.info(f"🥗 Recommended Diet: {predicted_diet}")
    else:
        st.warning("No diet recommendation found for this disease.")

    # Yoga
    if disease in yoga_map:
        st.subheader("🧘 Yoga Recommendations")
        for y in yoga_map[disease]:
            st.markdown(f"- **{y['Recommended Yoga Practice']}** ({y['Frequency']})")
            st.caption(f"Duration: {y['Estimated Duration to Benefit']} | Benefit: {y['Observed Benefit']}")
    else:
        st.warning("No yoga practices found for this condition.")
