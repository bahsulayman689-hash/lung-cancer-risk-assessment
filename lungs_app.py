#---------------------- lung_app.py---------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------------------------------------------
# 1. PAGE SETUP & GREEN SIDEBAR THEME
# --------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="🫁 Lung Cancer Predictor", 
    page_icon="🫁", 
    layout="wide"
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #2e7d32;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] hr {
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        .sidebar-btn {
            display: inline-block;
            padding: 6px 12px;
            background-color: rgba(255,255,255,0.15);
            color: white !important;
            border-radius: 4px;
            text-decoration: none;
            font-size: 14px;
            margin: 4px 2px;
            text-align: center;
        }
        .sidebar-btn:hover {
            background-color: rgba(255,255,255,0.25);
        }
    </style>
    """,
    unsafe_allow_html=True
)

header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("🫁 Lung Cancer Risk Assessment App")
    
    st.write("""
    Hi! I'm Sulayman Bah.
    I'm a machine learning and deep learning engineer.
    I build Machine Learning and
    Deep Learning applications
    using Python and Streamlit.
    """)
with header_col2:
    if os.path.exists("infectious-diseases.png"):
        st.image("infectious-diseases.png", width=200)
    else:
        st.info("💡 Logo asset not found.")
    
st.markdown("---")

# --------------------------------------------------------------------------------------------------
# 2. SIDEBAR CONTENT (LOGO, IMAGE & METADATA)
# --------------------------------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("infectious-diseases.png"):
        st.image("infectious-diseases.png", use_container_width=True)
    else:
        st.subheader("🟢 System Navigation")
        st.caption("(Tip: Add a 'infectious-diseases.png' image file to your folder)")
    
    st.markdown("---")
    
    if os.path.exists("IMG-20260704-WA0633.jpg"):
        st.image("IMG-20260704-WA0633.jpg", caption="App Operator Profile", use_container_width=True)
    else:
        st.markdown("👤 **Profile Avatar Place-holder**")
        st.caption("(Tip: Save your picture as 'IMG-20260704-WA0629.jpg' in this folder)")
        
    # 💼 Dynamic Social Anchors
    st.markdown(
        """
        <a class="sidebar-btn" href="https://github.com" target="_blank">🐙 GitHub</a>
        <a class="sidebar-btn" href="https://linkedin.com" target="_blank">💼 LinkedIn</a>
        <a class="sidebar-btn" href="mailto:bahsulayman689@gmail.com">📧 Contact</a>
        """, 
        unsafe_allow_html=True
    )
    
    # Who Am I Section
    st.markdown("### 🧑‍💻 Who Am I")
    st.markdown("""
    * **Name:** Sulayman Bah
    * **Role:** Machine Learning Engineer / Developer
    * **Focus:** Data science, predictive analytics, and building intelligent web applications.
    """)
    
    st.markdown("---")
    
    # My Skills Section
    st.markdown("### 🛠️ My Skills")
    st.markdown("""
    * 🐍 **Python Programming**
    * 📊 **Data Science & EDA** (Pandas, NumPy)
    * 🤖 **Machine Learning** (Scikit-Learn, XGBoost)
    * 📉 **Data Visualization** (Seaborn, Matplotlib)
    * 🖥️ **Web App Development** (Streamlit)
    * 🧠 **Model Deployment & Pipelines**
    """)

    st.markdown("---")
    st.markdown("### 📊 App Overview")
    st.write("This intelligence dashboard utilizes a trained XGBoost Classifier to compute lung cancer clinical risk scores based on physiological symptoms and lifestyle factors.")

# --------------------------------------------------------------------------------------------------
# 3. MAIN DASHBOARD CONTENT & PIPELINE LOADER
# --------------------------------------------------------------------------------------------------
st.title("🫁 Clinical Patient Screening Panel")
st.write("Input patient diagnostic attributes and symptom reports to analyze screening risk probabilities.")
st.markdown("---")

@st.cache_resource
def load_ml_pipeline():
    try:
        loaded_object = joblib.load("lungs_model.pkl")
        return loaded_object
    except FileNotFoundError:
        return "NOT_FOUND"

assets = load_ml_pipeline()

# CRITICAL PROTECTION CHECK: Verify if assets structure exists and matches expectations
if assets == "NOT_FOUND":
    st.error("🚨 **Error:** `lung_assets.pkl` not found in your directory!")
    st.info("Please run your updated `train.py` script first to save your model assets.")
    st.stop()
elif isinstance(assets, RandomForestClassifier):
    st.error("🚨 **Asset Format Mismatch Detected!**")
    st.warning("Your `lung_assets.pkl` only contains the model object. Remember to re-run your training script with the updated assets dictionary export code block.")
    st.stop()

# Extract variables safely once confirmed to be a dictionary structure
model = assets['model']
scaler = assets['scaler']
encoders = assets['encoders']
feature_columns = assets['feature_columns']

# --------------------------------------------------------------------------------------------------
# 4. INTERACTIVE DIAGNOSTIC INPUT FORMS
# --------------------------------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Patient Age:", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender Identification:", ["M", "F"])
    smoking = st.selectbox("History of Smoking?", ["YES", "NO"])
    yellow_fingers = st.selectbox("Presence of Yellow Fingers?", ["YES", "NO"])
    anxiety = st.selectbox("Exhibits Clinical Anxiety?", ["YES", "NO"])
    peer_pressure = st.selectbox("Subject to Peer Pressure?", ["YES", "NO"])
    chronic_disease = st.selectbox("History of Chronic Disease?", ["YES", "NO"])
    fatigue = st.selectbox("Experiences Chronic Fatigue?", ["YES", "NO"])

with col2:
    allergy = st.selectbox("Known Chronic Allergies?", ["YES", "NO"])
    wheezing = st.selectbox("Displays Audible Wheezing?", ["YES", "NO"])
    alcohol_consumption = st.selectbox("Consumes Alcohol regularly?", ["YES", "NO"])
    coughing = st.selectbox("Presents Frequent Coughing?", ["YES", "NO"])
    shortness_of_breath = st.selectbox("Experiences Shortness of Breath?", ["YES", "NO"])
    swallowing_difficulty = st.selectbox("Experiences Swallowing Difficulty?", ["YES", "NO"])
    chest_pain = st.selectbox("Reports Frequent Chest Pain?", ["YES", "NO"])

# --------------------------------------------------------------------------------------------------
# 5. XGBOOST INFERENCE & RISK ASSESSMENT ENGINE
# --------------------------------------------------------------------------------------------------
st.markdown("---")

if st.button("Generate Diagnostic Risk Report", type="primary", use_container_width=True):
    
    # Map input text strings down to match your exact numeric model keys (1=NO, 2=YES format)
    clinical_map = {"YES": "2", "NO": "1"}
    
    # Pass matching string representations down to encoders.transform() safely
        # Pass matching string representations down to encoders.transform() safely
    input_dict = {
        'GENDER': encoders['GENDER'].transform([gender])[0],
        'AGE': age,
        'SMOKING': encoders['SMOKING'].transform([clinical_map[smoking]])[0],
        'YELLOW_FINGERS': encoders['YELLOW_FINGERS'].transform([clinical_map[yellow_fingers]])[0],
        'ANXIETY': encoders['ANXIETY'].transform([clinical_map[anxiety]])[0],
        'PEER_PRESSURE': encoders['PEER_PRESSURE'].transform([clinical_map[peer_pressure]])[0],
        'CHRONIC DISEASE': encoders['CHRONIC DISEASE'].transform([clinical_map[chronic_disease]])[0],
        'FATIGUE ': encoders['FATIGUE '].transform([clinical_map[fatigue]])[0],
        'ALLERGY ': encoders['ALLERGY '].transform([clinical_map[allergy]])[0],
        'WHEEZING': encoders['WHEEZING'].transform([clinical_map[wheezing]])[0],
        'ALCOHOL CONSUMING': encoders['ALCOHOL CONSUMING'].transform([clinical_map[alcohol_consumption]])[0],
        'COUGHING': encoders['COUGHING'].transform([clinical_map[coughing]])[0],
        'SHORTNESS OF BREATH': encoders['SHORTNESS OF BREATH'].transform([clinical_map[shortness_of_breath]])[0],
        'SWALLOWING DIFFICULTY': encoders['SWALLOWING DIFFICULTY'].transform([clinical_map[swallowing_difficulty]])[0],
        'CHEST PAIN': encoders['CHEST PAIN'].transform([clinical_map[chest_pain]])[0]
    }
    
    # Inject the engineered interaction term required by your StandardScaler matrix
    input_dict['ANXYELFIN'] = input_dict['ANXIETY'] * input_dict['YELLOW_FINGERS']
    
    # Convert input array configuration directly into a Pandas DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Lock exact training data column order tracking sequence to prevent misalignment
    input_df = input_df[feature_columns]
    
    # Transform input scaling arrays safely
    input_scaled = scaler.transform(input_df)
    
    # Run structural prediction probabilities using your model 
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    
    # ----------------------------------------------------------------------------------------------
    # 6. METRICS DISPLAY WINDOWS
    # ----------------------------------------------------------------------------------------------
    st.subheader("📊 Diagnostic Assessment Results")
    
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.metric(label="Risk Probability Score", value=f"{prediction_proba * 100:.2f}%")
        
    with metric_col2:
        risk_status = "🔴 HIGH RISK" if prediction_proba >= 0.50 else "🟢 LOW RISK"
        st.metric(label="Clinical Categorization", value=risk_status)
        
    if prediction_proba >= 0.50:
        st.error("⚠️ **Notice:** High-risk indicators match historical positive baseline sets. Urgent specialist review is advised.")
    else:
        st.success("✅ **Notice:** Low matching footprint calculated relative to background target thresholds.")
