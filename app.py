import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt

# --- Import SHAP ---
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Columbia Asia | Cardiovascular Risk Assessment Prototype",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI ENHANCEMENT: Sidebar for Branding and Navigation ---
with st.sidebar:
    if os.path.exists('columbia.jpg'):
        st.image('columbia.jpg', width="stretch")
    st.title("Cardiovascular Risk Assessment Tool (Prototype)")
    st.markdown("---")
    st.header("⚠️ Important Disclaimer")
    st.warning(
        """
        This tool provides a risk estimation based on a machine learning model and is for **informational and demonstration purposes only**. 

        It is **not a substitute for professional medical diagnosis** or advice. Please consult with a qualified healthcare provider for any health concerns.
        """
    )
    st.markdown("---")
    st.info(
        """
        **About Columbia Hospital**\n
        Columbia Hospital is a leader in patient-centric healthcare, committed to leveraging technology to improve patient outcomes. 
        This prototype is part of our initiative to explore innovative health-tech solutions.
        """
    )

# Global variables to track library availability
LIGHTGBM_AVAILABLE = False
SKLEARN_AVAILABLE = False

# Check for required libraries
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    import sklearn
    from sklearn.preprocessing import MinMaxScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# Top 10 features
TOP_10_FEATURES = [
    'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
    'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level',
    'Alcohol Consumption'
]

# Numeric columns that need scaling
NUMERIC_COLS = [
    'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
    'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
]


@st.cache_resource
def load_assets():
    """Load the trained model, scaler, and SHAP explainer"""
    if not all([LIGHTGBM_AVAILABLE, SKLEARN_AVAILABLE, SHAP_AVAILABLE]):
        return None, None, None, "A required library is missing."

    try:
        if not os.path.exists('lgb.pkl'):
            return None, None, None, "Model file 'lgb.pkl' not found"
        if not os.path.exists('minmax_scaler.joblib'):
            return None, None, None, "Scaler file 'minmax_scaler.joblib' not found"

        model = joblib.load('lgb.pkl')
        scaler = joblib.load('minmax_scaler.joblib')
        explainer = shap.TreeExplainer(model)

        return model, scaler, explainer, "success"

    except Exception as e:
        return None, None, None, f"Error loading files: {str(e)}"


def make_prediction(user_data, model, scaler):
    """Make prediction using the loaded model"""
    try:
        df = pd.DataFrame([user_data])
        df_scaled = df.copy()
        df_scaled[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
        df_final = df_scaled[TOP_10_FEATURES]

        prediction = model.predict(df_final)[0]
        probability = model.predict_proba(df_final)[0]

        heart_disease_prob = probability[1]
        if heart_disease_prob < 0.3:
            risk_level = "Low"
        elif heart_disease_prob < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return prediction, probability, risk_level, df_final

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None, None


def display_results(user_data, prediction, probability, risk_level, df_final, explainer):
    """Display prediction results, SHAP explanations, and recommendations"""
    if prediction is None:
        return

    st.markdown("## 📊 Patient Risk Profile")

    col1, col2 = st.columns([1, 1])
    with col1:
        heart_disease_prob = probability[1]

        # --- UI ENHANCEMENT: Polished results cards ---
        if prediction == 1:
            st.markdown(
                f"""
                <div style="background-color: #FFF3F3; padding: 20px; border-radius: 10px; border-left: 6px solid #D9534F;">
                    <h3 style="color: #D9534F;">Risk Status: Elevated Risk Detected</h3>
                    <p style="font-size: 18px; margin: 10px 0;">
                        <strong>Risk Level: {risk_level.upper()}</strong><br>
                        Predicted Probability: <strong>{heart_disease_prob:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style="background-color: #F3FFF3; padding: 20px; border-radius: 10px; border-left: 6px solid #5CB85C;">
                    <h3 style="color: #5CB85C;">Risk Status: Low Risk Detected</h3>
                    <p style="font-size: 18px; margin: 10px 0;">
                        <strong>Risk Level: {risk_level.upper()}</strong><br>
                        Predicted Probability: <strong>{heart_disease_prob:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=heart_disease_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk Score"},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#333"},
                'steps': [
                    {'range': [0, 30], 'color': "#5CB85C"},
                    {'range': [30, 70], 'color': "#F0AD4E"},
                    {'range': [70, 100], 'color': "#D9534F"}],
            }))
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), font={'family': "Arial, sans-serif"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🔬 Analysis of Risk Factors")
    with st.expander("View detailed contribution of each health metric", expanded=True):
        st.info(
            "SHAP’s f(x) represents the model’s raw score (log‑odds), while the probability shown in the 'Patient Risk Profile' is the sigmoid‑transformed version of that score due to the LightGBM model, "
            "**they are the same prediction expressed in different forms**. ", icon='ℹ️'
        )
        shap_values = explainer.shap_values(df_final)

        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_for_positive_class = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_values_for_positive_class = shap_values
            expected_value = explainer.expected_value

        df_display = df_final.copy()
        for col in df_display.columns:
            if col in user_data:
                if col == 'Alcohol Consumption':
                    reverse_alcohol_map = {0: 'None', 1: 'Low', 2: 'Medium', 3: 'High'}
                    df_display[col] = reverse_alcohol_map.get(user_data[col], user_data[col])
                else:
                    df_display[col] = user_data[col]

        shap_explanation = shap.Explanation(
            values=shap_values_for_positive_class[0],
            base_values=expected_value, data=df_display.iloc[0],
            feature_names=df_final.columns.tolist()
        )

        fig = plt.figure(figsize=(1,2))
        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
        # plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.info(
            """
            **How to read this chart:**
            - The **base value** `E[f(x)]` is the model's average prediction over the entire dataset.
            - **Red bars** represent features that pushed the prediction **higher** (increased risk).
            - **Blue bars** represent features that pushed the prediction **lower** (decreased risk).
            - The length of each bar shows the magnitude of that feature's impact on this specific prediction.
            - `f(x)` at the top is the final output score for your inputs.
            """
        )

    display_recommendations(user_data, risk_level)
    st.markdown("---")


def display_recommendations(user_data, risk_level):
    """Display tailored recommendations based on risk level and inputs."""
    st.markdown("### 💡 Recommendations and Next Steps")

    key_factors = []
    if user_data['Blood Pressure'] >= 130: key_factors.append("high Blood Pressure")
    if user_data['Cholesterol Level'] >= 200: key_factors.append("high Cholesterol")
    if user_data['BMI'] >= 25: key_factors.append("high BMI")
    if user_data['Fasting Blood Sugar'] >= 100: key_factors.append("high Fasting Blood Sugar")
    if user_data['Alcohol Consumption'] > 1: key_factors.append("Alcohol Consumption")

    st.info(
        "Maintaining a balanced diet, regular physical activity (at least 150 minutes per week), and managing stress are beneficial for all risk levels.",
    icon='🩷'
    )

    # --- UI ENHANCEMENT: Hospital-specific Call to Action ---
    if risk_level == "High":
        st.error(
            f"**Urgent Consultation Recommended:** Your profile indicates a high risk. It is crucial to consult a cardiologist soon. Key areas of concern from your inputs include: **{', '.join(key_factors)}**.")
        st.link_button(
            icon="📅",
            label="**Schedule a wellness check-up at Columbia Hospital.**",
            url="https://www.columbiaasia.com/malaysia/make-an-appointment/",  # Placeholder URL
            width="content"
        )
    elif risk_level == "Moderate":
        st.warning(
            f"**Proactive Management Advised:** Your profile indicates a moderate risk. A consultation with a healthcare provider is recommended to discuss preventative strategies, especially regarding: **{', '.join(key_factors)}**.")
        st.link_button(
            icon="📅",
            label="**Schedule a wellness check-up at Columbia Hospital.**",
            url="https://www.columbiaasia.com/malaysia/make-an-appointment/",  # Placeholder URL
            width="content"
        )
    else:
        st.success(
            "**Continue Healthy Habits:** Your profile indicates a low risk. We commend your efforts in maintaining a healthy lifestyle. Regular check-ups are still recommended to monitor your health status.")


def collect_user_inputs():
    """Collect user inputs from the main page UI."""
    st.markdown("## 📋 Patient Health Metrics")

    # --- UI ENHANCEMENT: Grouping inputs into collapsible expanders ---
    with st.expander("👤 Personal & Lifestyle Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            user_data = {}
            user_data['Age'] = st.slider("Age (years)", 18, 100, 45)
        with col2:
            user_data['BMI'] = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1)
        with col3:
            user_data['Sleep Hours'] = st.slider("Average Sleep (hours/night)", 3.0, 12.0, 7.0, 0.5)

        alcohol_map = {'None': 0, 'Low (1-2 drinks/week)': 1, 'Medium (3-5 drinks/week)': 2, 'High (5+ drinks/week)': 3}
        alcohol_choice = st.selectbox(
            "Alcohol Consumption", options=list(alcohol_map.keys()), index=1)
        user_data['Alcohol Consumption'] = alcohol_map[alcohol_choice]

    with st.expander("❤️ Cardiovascular & Blood Metrics", expanded=True):
        colA, colB, colC, colD = st.columns(4)
        with colA:
            user_data['Blood Pressure'] = st.slider("Systolic Blood Pressure (mmHg)", 80, 200, 120)
        with colB:
            user_data['Cholesterol Level'] = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
        with colC:
            user_data['Triglyceride Level'] = st.slider("Triglycerides (mg/dL)", 50, 500, 150)
        with colD:
            user_data['Fasting Blood Sugar'] = st.slider("Fasting Blood Sugar (mg/dL)", 70, 300, 100)

    with st.expander("🔬 Advanced Biomarkers", expanded=True):
        colX, colY = st.columns(2)
        with colX:
            user_data['CRP Level'] = st.slider("C-Reactive Protein (CRP) (mg/L)", 0.1, 10.0, 1.0, 0.1)
        with colY:
            user_data['Homocysteine Level'] = st.slider("Homocysteine (μmol/L)", 5.0, 50.0, 10.0, 0.1)

    return user_data


def main():
    st.title("Clinical Prototype: Heart Disease Prediction System")
    st.image("heart.png", width=500)

    missing_libs = [lib for lib, available in
                    [("lightgbm", LIGHTGBM_AVAILABLE), ("scikit-learn", SKLEARN_AVAILABLE), ("shap", SHAP_AVAILABLE)] if
                    not available]
    if missing_libs:
        st.error(f"⚠️ Required libraries are missing: **{', '.join(missing_libs)}**. Please install them.")
        return

    model, scaler, explainer, status = load_assets()
    if model is None:
        st.error(f"❌ {status}. Please ensure required files are in the directory.")
        return

    # --- UI ENHANCEMENT: New tab structure ---
    tab1, tab2, tab3 = st.tabs(["🔬 **Risk Assessment Tool**", "📈 **Model Performance**", "ℹ️ **About This Tool**"])

    with tab1:
        user_data = collect_user_inputs()
        st.markdown("---")
        if st.button("Analyze Patient Data", type="primary", use_container_width=True):
            with st.spinner("Running predictive analysis..."):
                prediction, probability, risk_level, df_final = make_prediction(user_data, model, scaler)
                if prediction is not None:
                    display_results(user_data, prediction, probability, risk_level, df_final, explainer)

    with tab2:
        st.header("About The Prediction Model")
        st.markdown("""
        This system uses a **LightGBM (Light Gradient Boosting Machine)** model to predict heart disease risk. 
        LightGBM is a high-performance algorithm known for its accuracy and speed.
        """)

        st.subheader("Why LightGBM?")
        st.markdown(
            "- **Accuracy**: It excels at finding complex patterns in health data.\n"
            "- **Speed**: It is highly efficient, providing near-instant predictions.\n"
            "- **Interpretability**: Combined with SHAP, we can understand *why* a prediction was made.")

        st.subheader("Performance Metrics")
        st.markdown("The model was rigorously trained and evaluated on a set of 10000 individuals. "
                    "Below are the key performance indicators.")

        cols = st.columns(3)
        images = {'classification.png': 'Evaluation Metrics', 'confusion.png': 'Confusion Matrix',
                  'rocauc.png': 'ROC-AUC Curve'}
        for i, (img_file, caption) in enumerate(images.items()):
            with cols[i]:
                if os.path.exists(img_file):
                    st.image(img_file, caption=caption, use_container_width=True)
                else:
                    st.warning(f"Image '{img_file}' not found.")

        # --- UI ENHANCEMENT: Added Limitations section ---
        st.subheader("Limitations")
        st.warning("""
        - **Generalization**: The model's performance is dependent on the data it was trained on and may not be equally accurate for all demographic groups.
        - **Not a Diagnostic Tool**: This model provides a risk score, not a diagnosis. Many other factors, including genetics and lifestyle, contribute to heart disease.
        """)

    with tab3:
        st.header("About This Prototype")
        st.markdown("""
        This interactive tool is a **proof-of-concept prototype** developed to demonstrate the potential of machine learning in preventive healthcare. 
        It aims to provide clinicians and patients with an intuitive way to assess cardiovascular risk based on standard health metrics.

        **Key Objectives:**
        - **Demonstrate a Functional Model**: Showcase the end-to-end workflow from data input to risk analysis.
        - **Facilitate Discussion**: Act as a catalyst for discussions between data science teams and clinical stakeholders.
        - **Gather Feedback**: Collect feedback on the tool's usability, clarity, and potential for integration into clinical workflows.

        This application is built using Streamlit, Python, and the LightGBM machine learning library.
        """)
        st.markdown("---")
        st.markdown("""
        **For questions or feedback, please contact the development team at:**

        - **General Inquiries:** Jin [tanjinyuan-wm23@student.tarc.edu.my]  
        - **Technical Support:** Patricia [patricialh-wg23@student.tarc.edu.my] 
        - **Data & Analytics:** Skye [hengtl-wm23@student.tarc.edu.my]
        - **Clinical Liaison:** Kai Xin [virginckx-wm23@student.tarc.edu.my]
        """)


if __name__ == "__main__":
    main()