import streamlit as st
import pandas as pd
# import numpy as np
import joblib
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt

# --- NEW: Import SHAP ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
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

# --- UPDATED: Top 10 features now include Alcohol Consumption ---
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
    if not LIGHTGBM_AVAILABLE:
        return None, None, None, "LightGBM library is missing"
    
    if not SKLEARN_AVAILABLE:
        return None, None, None, "Scikit-learn library is missing"
        
    if not SHAP_AVAILABLE:
        return None, None, None, "SHAP library is missing"

    try:
        if not os.path.exists('lgb.pkl'):
            return None, None, None, "Model file 'lgb.pkl' not found"
            
        if not os.path.exists('minmax_scaler.joblib'):
            return None, None, None, "Scaler file 'minmax_scaler.joblib' not found"
        
        # Load model and scaler
        model = joblib.load('lgb.pkl')
        scaler = joblib.load('minmax_scaler.joblib')
        
        # --- NEW: Create and cache the SHAP explainer ---
        explainer = shap.TreeExplainer(model)
        
        return model, scaler, explainer, "success"
        
    except Exception as e:
        return None, None, None, f"Error loading files: {str(e)}"

def make_prediction(user_data, model, scaler):
    """Make prediction using the loaded model"""
    try:
        # Create DataFrame from user data
        df = pd.DataFrame([user_data])
        
        # Scale numeric features
        df_scaled = df.copy()
        df_scaled[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
        
        # Ensure column order matches model's expectation
        df_final = df_scaled[TOP_10_FEATURES]
        
        # Make prediction
        prediction = model.predict(df_final)[0]
        probability = model.predict_proba(df_final)[0]
        
        # Determine risk level
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
    
    st.markdown("## 📊 Prediction Results")
    
    # --- Prediction and Gauge Chart ---
    col1, col2 = st.columns([1, 1])
    with col1:
        heart_disease_prob = probability[1]
        if prediction == 1:
            st.markdown(
                f"""
                <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336;">
                    <h3 style="color: #d32f2f;">⚠️ Heart Disease Risk Detected</h3>
                    <p style="font-size: 18px; margin: 10px 0;">
                        <strong>Risk Level: {risk_level}</strong><br>
                        Probability: {heart_disease_prob:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
                    <h3 style="color: #388e3c;">✅ Low Heart Disease Risk</h3>
                    <p style="font-size: 18px; margin: 10px 0;">
                        <strong>Risk Level: {risk_level}</strong><br>
                        Probability: {heart_disease_prob:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=heart_disease_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [None, 100]}, 'bar': {'color': "#d32f2f"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
            }))
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- UPDATED & FIXED: SHAP Waterfall Plot for Prediction Explanation ---
    st.markdown("### 🔬 Model Prediction Breakdown")
    with st.expander("See how each factor contributed to the prediction", expanded=True):
        # Calculate SHAP values for the specific prediction
        shap_values = explainer.shap_values(df_final)
        
        # --- ROBUSTNESS FIX STARTS HERE ---
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_for_positive_class = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_values_for_positive_class = shap_values
            expected_value = explainer.expected_value
        # --- ROBUSTNESS FIX ENDS HERE ---
        
        # --- NEW: Create a clone DataFrame for display with the original user values ---
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
            base_values=expected_value,
            data=df_display.iloc[0],
            feature_names=df_final.columns.tolist()
        )
        
        fig, ax = plt.subplots(figsize=(5, 6))
        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
        plt.tight_layout()
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
    if user_data['Blood Pressure'] >= 130: key_factors.append("Blood Pressure")
    if user_data['Cholesterol Level'] >= 200: key_factors.append("Cholesterol Level")
    if user_data['BMI'] >= 25: key_factors.append("BMI")
    if user_data['Fasting Blood Sugar'] >= 100: key_factors.append("Fasting Blood Sugar")
    if user_data['Alcohol Consumption'] > 1: key_factors.append("Alcohol Consumption")

    if risk_level == "High":
        st.error(f"**Urgent Action Recommended:** Your results indicate a high risk. It is crucial to consult a healthcare professional soon to discuss these results, especially your levels for: **{', '.join(key_factors)}**.")
    elif risk_level == "Moderate":
        st.warning(f"**Proactive Management Advised:** Your results indicate a moderate risk. This is an important opportunity to make positive lifestyle changes. Consider discussing your **{', '.join(key_factors)}** with a doctor.")
    else:
        st.success("**Continue a Healthy Lifestyle:** Your results indicate a low risk, which is excellent! Continue your healthy habits to maintain this status.")

def collect_user_inputs():
    """Collect user inputs from the main page UI."""
    st.markdown("## 📋 Patient Information")
    st.markdown("Please fill in the following health metrics:")
    
    user_data = {}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Basic Information")
        user_data['Age'] = st.slider("Age (years)", 18, 100, 45)
        user_data['BMI'] = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1)
        user_data['Sleep Hours'] = st.slider("Sleep Hours per night", 3.0, 12.0, 7.0, 0.5)
        
        alcohol_map = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}
        alcohol_choice = st.selectbox(
            "Alcohol Consumption", 
            options=['None', 'Low', 'Medium', 'High'], index=1,
            help="Average alcohol consumption level")
        user_data['Alcohol Consumption'] = alcohol_map[alcohol_choice]
        
    with col2:
        st.markdown("#### Cardiovascular Metrics")
        user_data['Blood Pressure'] = st.slider("Systolic Blood Pressure (mmHg)", 80, 200, 120)
        user_data['Cholesterol Level'] = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
        user_data['Triglyceride Level'] = st.slider("Triglycerides (mg/dL)", 50, 500, 150)
        user_data['Fasting Blood Sugar'] = st.slider("Fasting Blood Sugar (mg/dL)", 70, 300, 100)
        
    with col3:
        st.markdown("#### Biomarkers")
        user_data['CRP Level'] = st.slider("CRP Level (mg/L)", 0.1, 10.0, 1.0, 0.1)
        user_data['Homocysteine Level'] = st.slider("Homocysteine (μmol/L)", 5.0, 50.0, 10.0, 0.1)
        
    return user_data

# --- NEW: main() function is completely restructured for tabs ---
def main():
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("An interactive tool to predict heart disease risk based on key health metrics.")
    st.warning("⚠️ **Disclaimer**: This is an educational tool, not a substitute for professional medical advice.")
    st.markdown("---")
    
    # Library and asset loading checks
    missing_libs = []
    if not LIGHTGBM_AVAILABLE: missing_libs.append("lightgbm")
    if not SKLEARN_AVAILABLE: missing_libs.append("scikit-learn")
    if not SHAP_AVAILABLE: missing_libs.append("shap")

    if missing_libs:
        st.error(f"⚠️ The following libraries are missing: **{', '.join(missing_libs)}**.")
        st.markdown("## 🔧 Installation Required")
        for lib in missing_libs:
            st.code(f"pip install {lib}", language="bash")
        st.markdown("After installation, please refresh the page.")
        return

    model, scaler, explainer, status = load_assets()
    
    if model is None:
        st.error(f"❌ {status}")
        st.markdown("Please ensure `lgb.pkl` and `minmax_scaler.joblib` are in the same directory.")
        return
    
    # --- Create the tabs ---
    tab1, tab2 = st.tabs(["🔬 **Prediction Tool**", "📊 **Model Overview**"])

    # --- Content for the Prediction Tool Tab ---
    with tab1:
        # We need to notify the user that the model is ready inside the tab
        st.success("✅ Model, scaler, and explainer loaded successfully and ready!")
        
        user_data = collect_user_inputs()
        
        st.markdown("---")
        if st.button("🔬 Predict Heart Disease Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing your health data..."):
                prediction, probability, risk_level, df_final = make_prediction(user_data, model, scaler)
                if prediction is not None:
                    display_results(user_data, prediction, probability, risk_level, df_final, explainer)

    # --- Content for the Model Overview Tab ---
    with tab2:
        st.subheader("About The Prediction Model")
        st.markdown("""
        This application uses a **LightGBM (Light Gradient Boosting Machine)** model, a high-performance gradient boosting framework, to predict the risk of heart disease. The model was trained on a dataset of health metrics to identify patterns that may indicate cardiovascular risk.

        LightGBM is known for its speed and efficiency, making it well-suited for this type of prediction task.
        """)
        
        st.subheader("Model Performance")
        st.markdown("The model's performance was evaluated using several standard metrics to ensure its reliability and accuracy. The image below summarizes the evaluation results on the test dataset.")
        
        width = 450
        col1, col2, col3 = st.columns(3)
        # Display the evaluation image
        with col1:
            if os.path.exists('classification.png'):
                st.image('classification.png', caption='Evaluation Metrics', width=width)
            else:
                st.warning("Evaluation image 'classification.png' not found.")

        # Column 2: Confusion Matrix
        with col2:
            if os.path.exists('confusion.png'):
                st.image('confusion.png', caption='LightGBM Model Confusion Matrix', width=width)
            else:
                st.warning("Evaluation image 'confusion.png' not found.")

        # Column 3: ROC-AUC Curve
        with col3:
            if os.path.exists('rocauc.png'):
                st.image('rocauc.png', caption='LightGBM Model ROC-AUC Curve',  width=width)
            else:
                st.warning("Evaluation image 'rocauc.png' not found.")

if __name__ == "__main__":
    main()