import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests

# ---------------- LOAD MODEL ----------------
model = joblib.load("salary_prediction_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SalaryAI - Predict Your Worth", page_icon="💼", layout="wide")

# ---------------- LOAD ANIMATION ----------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

money_anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_0fhlytwe.json")

# ---------------- MODERN CUSTOM CSS ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    .main {
        background: transparent !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* --- Hero Section --- */
    .hero-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 85vh;
        padding: 2rem;
    }

    .hero-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2rem;
        max-width: 700px;
        background: rgba(255, 255, 255, 0.95);
        padding: 4rem 3rem;
        border-radius: 25px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }

    .title-text {
        font-size: 68px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .subtitle-text {
        color: #6b7280;
        font-size: 18px;
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* --- Button Styling --- */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px 50px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        margin-top: 1rem !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.6) !important;
    }

    .stButton>button:active {
        transform: translateY(-1px) !important;
    }

    /* --- Form Container --- */
    .form-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }

    .form-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
    }

    .input-label {
        color: #374151;
        font-weight: 600;
        font-size: 14px;
    }

    /* --- Back Button --- */
    .back-btn-style {
        position: fixed;
        top: 25px;
        left: 30px;
        font-size: 32px;
        cursor: pointer;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        z-index: 999;
    }

    .back-btn-style:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transform: scale(1.1);
        color: white;
    }

    /* --- Result Card --- */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
    }

    .result-amount {
        font-size: 48px;
        font-weight: 800;
        margin: 1rem 0;
        letter-spacing: -1px;
    }

    .success-message {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* --- Streamlit Input Override --- */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #e5e7eb !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- HOME PAGE ----------------
if st.session_state.page == "home":
    st.markdown("""
        <div class="hero-wrapper">
            <div class="hero-content">
                <h1 class="title-text">SalaryAI</h1>
                <p class="subtitle-text">
                    💡 Predict your market value with AI precision<br>
                    Based on experience, education, location & more
                </p>
    """, unsafe_allow_html=True)

    # Display animation if loaded
    if money_anim:
        st_lottie(money_anim, speed=1, height=280, key="home_anim")
    else:
        st.markdown("<div style='height: 280px; display: flex; align-items: center; justify-content: center;'><p style='color: #999;'>✨ Loading animation...</p></div>", unsafe_allow_html=True)

    # Get Started Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True, key="get_started_btn"):
            st.session_state.page = "predict"
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)




# ---------------- PREDICTION PAGE ----------------
elif st.session_state.page == "predict":
    # Back button
    col1, col2 = st.columns([1, 20])
    with col1:
        if st.button("⬅️", key="back_btn", help="Go back to home"):
            st.session_state.page = "home"
            st.rerun()

    # Form container
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="form-title">📋 Employee Profile</h2>', unsafe_allow_html=True)

    # Create two-column layout for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="input-label">Basic Information</p>', unsafe_allow_html=True)
        age = st.number_input("👤 Age", min_value=18, max_value=70, value=30, step=1)
        education_map = {1: "Bachelor's Degree", 2: "Master's Degree", 3: "Ph.D."}
        education = st.selectbox("🎓 Education Level", [1, 2, 3], format_func=lambda x: education_map[x], index=1)
        experience = st.slider("📈 Years of Experience", min_value=0.0, max_value=50.0, value=3.0, step=0.5)

    with col2:
        st.markdown('<p class="input-label">Professional Details</p>', unsafe_allow_html=True)
        gender = st.selectbox("👥 Gender", label_encoders["Gender"].classes_.tolist())
        job = st.selectbox("💼 Job Title", label_encoders["Job Title"].classes_.tolist())
        country = st.selectbox("🌍 Country", label_encoders["Country"].classes_.tolist())
    
    # Third row - additional info
    col3, col4 = st.columns(2)
    with col3:
        race = st.selectbox("🎯 Race/Ethnicity", label_encoders["Race"].classes_.tolist())
    with col4:
        senior_map = {0: "No", 1: "Yes"}
        senior = st.selectbox("⭐ Senior Position?", [0, 1], format_func=lambda x: senior_map[x])

    st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if st.button("💰 Predict Salary", use_container_width=True, key="predict_btn"):
            try:
                df_input = pd.DataFrame([{
                    "Age": age,
                    "Gender": gender,
                    "Education Level": education,
                    "Job Title": job,
                    "Years of Experience": experience,
                    "Country": country,
                    "Race": race,
                    "Senior": senior
                }])

                for col in ["Gender", "Job Title", "Country", "Race"]:
                    le = label_encoders[col]
                    df_input[col] = le.transform(df_input[col])

                order = ["Age", "Gender", "Education Level", "Job Title",
                         "Years of Experience", "Country", "Race", "Senior"]
                df_input = df_input[order]

                scaled = scaler.transform(df_input)
                prediction = model.predict(scaled)[0]

                # Display result
                st.markdown(f"""
                    <div class="result-card">
                        <div class="success-message">✨ Your Predicted Salary</div>
                        <div class="result-amount">${prediction:,.0f}</div>
                        <p style="font-size: 14px; margin-top: 1rem; opacity: 0.95;">
                            Based on your profile: {job} with {experience} years of experience in {country}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.balloons()
                
                # Show animation
                if money_anim:
                    st_lottie(money_anim, speed=1.2, height=250, key="result_anim")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your inputs and try again.")
