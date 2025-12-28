import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import requests
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler # Required for pipelines

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Medical Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL & DATA CONFIGURATION ---
MODEL_DIR = './models/'
FEVER_THRESHOLD = 38.0
LOOKBACK_WINDOW = pd.Timedelta(minutes=5)

# --- THINGSPeAK CONFIGURATION ---
THINGSPEAK_CHANNEL_ID = "3110381"
THINGSPEAK_READ_API_KEY = "X998UI7V7LF1X3Z9" # Use Read API Key
FIELD_MAP = {
    'field1': 'heart_rate', 'field2': 'spo2', 'field3': 'activity',
    'field4': 'sound_level', 'field5': 'temperature'
}

# --- SESSION STATE INITIALIZATION ---
# This is used to pass data from the input column to the results column
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# --- LOAD MODELS ---
@st.cache_resource
def load_model(path):
    """Loads the joblib model file."""
    if not os.path.exists(path):
        # Don't show an error here, let the prediction button handle it
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        # Log error to console, but don't show on-screen until needed
        print(f"Error loading model {os.path.basename(path)}: {e}")
        return None

# Load all four models
diabetes_model = load_model(os.path.join(MODEL_DIR, 'diabetes_prediction_model.joblib'))
hypertension_model = load_model(os.path.join(MODEL_DIR, 'hypertension_model.joblib'))
ckd_model = load_model(os.path.join(MODEL_DIR, 'ckd_model.joblib'))
fever_model = load_model(os.path.join(MODEL_DIR, 'fever_model.joblib'))


# --- LIVE MONITOR FUNCTIONS ---
@st.cache_data(ttl=55) # Cache data for just under the 1-min refresh
def fetch_live_data():
    """Fetches the latest data window from ThingSpeak."""
    num_records = int(LOOKBACK_WINDOW.total_seconds() / 15) + 5 # Assuming 15s interval
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"
    params = {"api_key": THINGSPEAK_READ_API_KEY, "results": num_records}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()['feeds']
        if len(data) < 2: return None
        
        live_df = pd.DataFrame(data).rename(columns=FIELD_MAP)
        live_df['created_at'] = pd.to_datetime(live_df['created_at'])
        live_df = live_df.set_index('created_at')
        for col in FIELD_MAP.values():
            if col in live_df.columns:
                live_df[col] = pd.to_numeric(live_df[col], errors='coerce')
        # Use ffill (forward fill) and then dropna to handle missing sensor reads
        return live_df.ffill().dropna() 
    except Exception as e:
        # Silently fail for the loop, but log to console
        print(f"Error fetching ThingSpeak data: {e}")
        return None

def create_live_features(df):
    """Creates time-series features from a live data window."""
    if df is None or len(df) < 2: return None
    features = {}
    time_diff = (df.index - df.index[0]).total_seconds()
    
    # Calculate features for all sensors used in the fever model
    # This must match the features created in train_fever_model.py
    for sensor, prefix in [('temperature', 'temp_'), ('heart_rate', 'hr_'), 
                           ('spo2', 'spo2_'), ('activity', 'activity_'), 
                           ('sound_level', 'sound_level_')]:
        if sensor in df.columns:
            features[f'{prefix}mean'] = df[sensor].mean()
            if prefix in ['temp_', 'hr_']: # Only temp/hr have std and trend
                features[f'{prefix}std'] = df[sensor].std()
                try:
                    features[f'{prefix}trend'] = np.polyfit(time_diff, df[sensor], 1)[0]
                except (np.linalg.LinAlgError, TypeError):
                    features[f'{prefix}trend'] = 0
    
    return pd.DataFrame([features]).fillna(0)


# --- WEB APP INTERFACE ---
st.title('🩺 AI Medical Monitor')

# --- GLOBAL SIDEBAR ---
# This sidebar is for global controls only (like 'About')
st.sidebar.title("About")
st.sidebar.info("This application combines static EHR risk prediction with real-time IoT vital sign monitoring.")

# --- CREATE TABS ---
tab1, tab2 = st.tabs(["EHR Risk Predictor", "Live Patient Monitor"])

# ===================================================================
# TAB 1: EHR RISK PREDICTOR
# ===================================================================
with tab1:
    st.header("EHR-Based Condition Risk Predictor")
    st.markdown("Select a condition, input patient data, and click 'Predict' to see the risk analysis.")
    
    # Create two columns for layout: 1 for inputs, 1 for results
    col1, col2 = st.columns([1, 1]) # 1:1 ratio

    # --- Column 1: Inputs ---
    with col1:
        st.subheader('Patient EHR Data')
        
        # This selectbox now lives in the main column
        prediction_choice = st.selectbox(
            'Choose a condition to predict:',
            ['Type-2 Diabetes', 'Hypertension', 'Chronic Kidney Disease'],
            key='prediction_choice'
        )

        # --- DIABETES MODEL INPUTS ---
        if prediction_choice == 'Type-2 Diabetes':
            st.subheader('Input for Diabetes Prediction')
            age = st.number_input('Age', 1, 120, 55, key="d_age")
            bmi = st.number_input('Body Mass Index (BMI)', 10.0, 60.0, 32.5, format="%.1f", key="d_bmi")
            systolic_bp = st.number_input('Systolic BP (mmHg)', 80, 250, 145, key="d_sys")
            diastolic_bp = st.number_input('Diastolic BP (mmHg)', 40, 150, 92, key="d_dia")
            total_cholesterol = st.number_input('Total Cholesterol (mg/dL)', 100, 400, 210, key="d_chol")
            smoking_status = st.selectbox('Smoking Status', ['non-smoker', 'former-smoker', 'smoker'], key="d_smoke")

            if st.button('Predict Diabetes Risk', type="primary", key="d_button"):
                if diabetes_model is None:
                    st.session_state.prediction_result = {'error': 'Diabetes model (diabetes_prediction_model.joblib) not found.'}
                else:
                    feature_data = {'AGE': [age], 'BMI': [bmi], 'SYSTOLIC_BP': [systolic_bp], 'DIASTOLIC_BP': [diastolic_bp], 'TOTAL_CHOLESTEROL': [total_cholesterol], 'SMOKING_STATUS': [smoking_status]}
                    input_df = pd.DataFrame(feature_data)
                    prediction = diabetes_model.predict(input_df)[0]
                    prediction_proba = diabetes_model.predict_proba(input_df)[0][1]
                    # Save results to session state
                    st.session_state.prediction_result = {
                        'type': 'Diabetes',
                        'prediction': prediction,
                        'proba': prediction_proba
                    }

        # --- HYPERTENSION MODEL INPUTS ---
        elif prediction_choice == 'Hypertension':
            st.subheader('Input for Hypertension Prediction')
            age = st.number_input('Age', 1, 120, 60, key="h_age")
            bmi = st.number_input('Body Mass Index (BMI)', 10.0, 60.0, 28.0, format="%.1f", key="h_bmi")
            systolic_bp = st.number_input('Systolic BP (mmHg)', 80, 250, 135, key="h_sys")
            diastolic_bp = st.number_input('Diastolic BP (mmHg)', 40, 150, 85, key="h_dia")
            total_cholesterol = st.number_input('Total Cholesterol (mg/dL)', 100, 400, 220, key="h_chol")
            smoking_status = st.selectbox('Smoking Status', ['non-smoker', 'former-smoker', 'smoker'], key="h_smoke")
            
            if st.button('Predict Hypertension Risk', type="primary", key="h_button"):
                if hypertension_model is None:
                    st.session_state.prediction_result = {'error': 'Hypertension model (hypertension_model.joblib) not found.'}
                else:
                    feature_data = {'AGE': [age], 'BMI': [bmi], 'SYSTOLIC_BP': [systolic_bp], 'DIASTOLIC_BP': [diastolic_bp], 'TOTAL_CHOLESTEROL': [total_cholesterol], 'SMOKING_STATUS': [smoking_status]}
                    input_df = pd.DataFrame(feature_data)
                    prediction = hypertension_model.predict(input_df)[0]
                    prediction_proba = hypertension_model.predict_proba(input_df)[0][1]
                    # Save results to session state
                    st.session_state.prediction_result = {
                        'type': 'Hypertension',
                        'prediction': prediction,
                        'proba': prediction_proba,
                        'what_if_data': input_df # Need to pass this for the what-if
                    }

        # --- CKD MODEL INPUTS ---
        elif prediction_choice == 'Chronic Kidney Disease':
            st.subheader('Input for CKD Prediction')
            age = st.number_input('Age', 1, 120, 65, key="c_age")
            bmi = st.number_input('Body Mass Index (BMI)', 10.0, 60.0, 29.5, format="%.1f", key="c_bmi")
            systolic_bp = st.number_input('Systolic BP (mmHg)', 80, 250, 150, key="c_sys")
            serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', 0.1, 15.0, 1.3, format="%.1f", key="c_serum")
            has_diabetes = st.selectbox('History of Diabetes?', ['No', 'Yes'], key="c_diab")
            has_hypertension = st.selectbox('History of Hypertension?', ['No', 'Yes'], key="c_hyper")
            
            if st.button('Predict CKD Risk', type="primary", key="c_button"):
                if ckd_model is None:
                    st.session_state.prediction_result = {'error': 'CKD model (ckd_model.joblib) not found.'}
                else:
                    feature_data = {'AGE': [age], 'BMI': [bmi], 'SYSTOLIC_BP': [systolic_bp], 'SERUM_CREATININE': [serum_creatinine], 'HAS_DIABETES': [1 if has_diabetes == 'Yes' else 0], 'HAS_HYPERTENSION': [1 if has_hypertension == 'Yes' else 0]}
                    input_df = pd.DataFrame(feature_data)
                    prediction = ckd_model.predict(input_df)[0]
                    prediction_proba = ckd_model.predict_proba(input_df)[0][1]
                    # Save results to session state
                    st.session_state.prediction_result = {
                        'type': 'CKD',
                        'prediction': prediction,
                        'proba': prediction_proba
                    }

    # --- Column 2: Results ---
    with col2:
        st.subheader('Prediction Result')
        
        # Check if a prediction has been made
        if st.session_state.prediction_result is None:
            st.info("Click a 'Predict' button on the left to see results here.")
        
        # Check for model loading errors
        elif 'error' in st.session_state.prediction_result:
            st.error(st.session_state.prediction_result['error'])
        
        # Display Diabetes Results
        elif st.session_state.prediction_result['type'] == 'Diabetes':
            result = st.session_state.prediction_result
            if result['prediction'] == 1:
                st.warning(f'High Risk of Type-2 Diabetes Detected')
                st.metric(label="Risk Score", value=f"{result['proba']:.2%}")
            else:
                st.success(f'Low Risk of Type-2 Diabetes Detected')
                st.metric(label="Confidence Score (Low Risk)", value=f"{1 - result['proba']:.2%}")
            st.info("Disclaimer: This is an AI-generated prediction based on a simulated dataset and should not be used for real medical diagnosis.")

        # Display Hypertension Results
        elif st.session_state.prediction_result['type'] == 'Hypertension':
            result = st.session_state.prediction_result
            if result['prediction'] == 1:
                st.warning(f'High Risk of Hypertension Detected')
                st.metric(label="Risk Score", value=f"{result['proba']:.2%}")
            else:
                st.success(f'Low Risk of Hypertension Detected')
                st.metric(label="Confidence Score (Low Risk)", value=f"{1 - result['proba']:.2%}")
            
            # --- What-If Analysis Section ---
            st.subheader("Risk Factor Analysis (What-If)")
            st.markdown("See how changing smoking status could impact this patient's risk score.")
            
            input_df = result['what_if_data']
            prediction_proba = result['proba']
            current_smoking_status = input_df['SMOKING_STATUS'].iloc[0]
            other_statuses = [s for s in ['non-smoker', 'former-smoker', 'smoker'] if s != current_smoking_status]

            for status in other_statuses:
                temp_df = input_df.copy()
                temp_df['SMOKING_STATUS'] = status
                what_if_proba = hypertension_model.predict_proba(temp_df)[0][1]
                delta = what_if_proba - prediction_proba
                st.metric(
                    label=f"Risk if status were '{status.replace('-', ' ').title()}'",
                    value=f"{what_if_proba:.2%}",
                    delta=f"{delta:.2%}",
                    delta_color="inverse"
                )
            st.info("Disclaimer: This is an AI-generated prediction based on a simulated dataset and should not be used for real medical diagnosis.")

        # Display CKD Results
        elif st.session_state.prediction_result['type'] == 'CKD':
            result = st.session_state.prediction_result
            if result['prediction'] == 1:
                st.warning(f'High Risk of Chronic Kidney Disease Detected')
                st.metric(label="Risk Score", value=f"{result['proba']:.2%}")
            else:
                st.success(f'Low Risk of Chronic Kidney Disease Detected')
                st.metric(label="Confidence Score (Low Risk)", value=f"{1 - result['proba']:.2%}")
            st.info("Disclaimer: This is an AI-generated prediction based on a simulated dataset and should not be used for real medical diagnosis.")


# ===================================================================
# TAB 2: LIVE PATIENT MONITOR
# ===================================================================
with tab2:
    st.header("🌡️ Live Fever Spike Prediction")
    
    if fever_model is None:
        st.error("Fever prediction model (fever_model.joblib) not found. Please run the training scripts first.")
    else:
        st.markdown("This section continuously fetches live sensor data to predict the risk of a fever spike (>38.0°C) in the next 15 minutes. The prediction auto-refreshes every minute.")
        
        # Create placeholder for the dynamic prediction content
        prediction_placeholder = st.empty()

    # --- This HTML is the "beautified" Chart.js dashboard ---
    st.header("Live Patient Vitals Dashboard")
    st.markdown("This dashboard pulls live data from the ThingSpeak server every 20 seconds.")

    # --- Use an f-string (f""") to inject Python variables ---
    # Note the double curly braces {{...}} for CSS rules
    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Custom Patient Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #333; margin: 0; padding: 0; }}
            h1 {{ display: none; }}
            .dashboard-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding-top: 20px; }}
            .chart-widget {{ background-color: #ffffff; border: 1px solid #d1d1d1; border-radius: 12px; box-shadow: 0 6px 16px rgba(0, 0, 0, 0.07); padding: 20px; width: 450px; box-sizing: border-box; }}
            .chart-widget h3 {{ margin-top: 0; margin-bottom: 15px; color: #3f51b5; border-bottom: 2px solid #eeeeee; padding-bottom: 10px; font-size: 1.25em; }}
            .chart-container {{ position: relative; height: 250px; }}
            .loading {{ text-align: center; padding-top: 80px; font-size: 1.2em; color: #888; }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="chart-widget">
                <h3>❤️ Heart Rate</h3>
                <div class="chart-container"><canvas id="hrChart"></canvas><div id="loading1" class="loading">Loading data...</div></div>
            </div>
            <div class="chart-widget">
                <h3>💨 SpO2</h3>
                <div class="chart-container"><canvas id="spo2Chart"></canvas><div id="loading2" class="loading">Loading data...</div></div>
            </div>
            <div class="chart-widget">
                <h3>🤸 Activity</h3>
                <div class="chart-container"><canvas id="gyroChart"></canvas><div id="loading3" class="loading">Loading data...</div></div>
            </div>
            <div class="chart-widget">
                <h3>🎤 Sound Level</h3>
                <div class="chart-container"><canvas id="soundChart"></canvas><div id="loading4" class="loading">Loading data...</div></div>
            </div>
            <div class="chart-widget">
                <h3>🌡️ Temperature</h3>
                <div class="chart-container"><canvas id="tempChart"></canvas><div id="loading5" class="loading">Loading data...</div></div>
            </div>
        </div>
        <script>
            // Inject Python variables directly into JavaScript
            const CHANNEL_ID = '{THINGSPEAK_CHANNEL_ID}';
            const READ_API_KEY = '{THINGSPEAK_READ_API_KEY}';
            const RESULTS_COUNT = 30; // Show last 30 data points
            const UPDATE_INTERVAL_MS = 20000; // 20 seconds
            
            // --- THIS IS THE CORRECTED LINE ---
            // Use ${{...}} to escape the braces for Python's f-string
            const API_URL = `https://api.thingspeak.com/channels/${{CHANNEL_ID}}/feeds.json?api_key=${{READ_API_KEY}}&results=${{RESULTS_COUNT}}`;
            
            let charts = {{}};

            function createChart(canvasId, label, color, yLabel) {{
                const ctx = document.getElementById(canvasId).getContext('2d');
                return new Chart(ctx, {{
                    type: 'line',
                    data: {{ labels: [], datasets: [{{ label: label, data: [], borderColor: color, backgroundColor: color + '33', borderWidth: 2.5, fill: true, tension: 0.4 }}] }},
                    options: {{
                        responsive: true, maintainAspectRatio: false,
                        scales: {{ x: {{ type: 'time', time: {{ tooltipFormat: 'HH:mm:ss' }}, ticks: {{ maxTicksLimit: 7 }} }}, y: {{ beginAtZero: false, title: {{ display: true, text: yLabel }} }} }},
                        plugins: {{ legend: {{ display: false }} }}
                    }}
                }});
            }}

            async function fetchAndUpdateCharts() {{
                try {{
                    const response = await fetch(API_URL);
                    if (!response.ok) throw new Error(`HTTP error! status: ${{response.status}}`);
                    const data = await response.json();
                    const feeds = data.feeds;
                    
                    // Hide all loaders
                    for (let i = 1; i <= 5; i++) {{
                        const loader = document.getElementById(`loading${{i}}`);
                        if (loader) loader.style.display = 'none';
                    }}

                    const labels = feeds.map(feed => new Date(feed.created_at));
                    const chartData = {{
                        hr: feeds.map(feed => feed.field1),
                        spo2: feeds.map(feed => feed.field2),
                        gyro: feeds.map(feed => feed.field3),
                        sound: feeds.map(feed => feed.field4),
                        temp: feeds.map(feed => feed.field5)
                    }};

                    // Update charts
                    charts.hr.data.labels = labels; charts.hr.data.datasets[0].data = chartData.hr; charts.hr.update();
                    charts.spo2.data.labels = labels; charts.spo2.data.datasets[0].data = chartData.spo2; charts.spo2.update();
                    charts.gyro.data.labels = labels; charts.gyro.data.datasets[0].data = chartData.gyro; charts.gyro.update();
                    charts.sound.data.labels = labels; charts.sound.data.datasets[0].data = chartData.sound; charts.sound.update();
                    charts.temp.data.labels = labels; charts.temp.data.datasets[0].data = chartData.temp; charts.temp.update();

                }} catch (error) {{ 
                    console.error("Error fetching ThingSpeak data:", error); 
                    // Optionally show an error on the chart widgets
                }}
            }}

            document.addEventListener('DOMContentLoaded', () => {{
                if (Object.keys(charts).length === 0) {{ // Only init once
                    charts.hr = createChart('hrChart', 'Heart Rate', '#d62020', 'BPM');
                    charts.spo2 = createChart('spo2Chart', 'SpO2', '#1976d2', '%');
                    charts.gyro = createChart('gyroChart', 'Activity', '#f57c00', 'rad/s');
                    charts.sound = createChart('soundChart', 'Sound Level', '#388e3c', 'Level');
                    charts.temp = createChart('tempChart', 'Temperature', '#7b1fa2', '°C');
                    
                    // Initial fetch
                    fetchAndUpdateCharts();
                    
                    // Set interval to refresh
                    setInterval(fetchAndUpdateCharts, UPDATE_INTERVAL_MS);
                }}
            }});
        </script>
    </body>
    </html>
    """

    # Embed the HTML in the app
    components.html(dashboard_html, height=800, scrolling=True)

    # --- Auto-refreshing loop for PREDICTION (must be at the end) ---
    if fever_model is not None:
        # This loop will run indefinitely
        while True: 
            live_df = fetch_live_data()

            # Use the placeholder to rewrite the prediction section
            with prediction_placeholder.container():
                if live_df is None or live_df.empty or len(live_df) < 2:
                    st.warning("Waiting for sufficient live data from ThingSpeak... Please ensure at least 5 minutes of data is available for prediction.")
                else:
                    # --- METRICS & PREDICTION ---
                    kpi1, kpi2, kpi3 = st.columns(3)
                    try:
                        # Display the *latest* values as KPIs
                        kpi1.metric(label="Latest Temperature (°C)", value=f"{live_df['temperature'].iloc[-1]:.2f}")
                        kpi2.metric(label="Latest Heart Rate (BPM)", value=f"{live_df['heart_rate'].iloc[-1]:.0f}")
                        kpi3.metric(label="Latest SpO₂ (%)", value=f"{live_df['spo2'].iloc[-1]:.1f}")
                    except (IndexError, KeyError):
                         st.warning("Data stream is missing one or more key fields (temp, hr, spo2).")

                    # Create features from the *entire* 5-min window
                    live_features = create_live_features(live_df)
                    
                    if live_features is not None:
                        try:
                            # Ensure column order matches the model's expectation
                            training_cols = fever_model.named_steps['scaler'].get_feature_names_out()
                            live_features = live_features[training_cols]
                            
                            prediction = fever_model.predict(live_features)[0]
                            prediction_proba = fever_model.predict_proba(live_features)[0][1]

                            st.subheader("Fever Spike Prediction (Next 15 Mins)")
                            if prediction == 1:
                                st.warning(f"High Risk of Fever Spike Detected")
                                st.metric(label="Risk Score", value=f"{prediction_proba:.2%}")
                            else:
                                st.success(f"Low Risk of Fever Spike Detected")
                                st.metric(label="Confidence Score (No Spike)", value=f"{1-prediction_proba:.2%}")
                        
                        except Exception as e:
                            st.error(f"Prediction error: Could not align features. Have you retrained the model? Error: {e}")
                    
                    st.subheader("Live Sensor Trends (Used for Prediction)")
                    # Chart only the features used in the model
                    if 'temperature' in live_df.columns and 'heart_rate' in live_df.columns:
                        chart_df = live_df[['temperature', 'heart_rate']].reset_index()
                        st.line_chart(chart_df, x='created_at', y=['temperature', 'heart_rate'])
            
            # Wait for 60 seconds before refreshing the prediction
            time.sleep(60)