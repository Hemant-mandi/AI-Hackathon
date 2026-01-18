import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import base64

# --- CONFIGURATION ---
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# --- CUSTOM CSS FOR PROFESSIONAL TYPOGRAPHY ---
st.markdown("""
    <style>
    /* Global Font Adjustments */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Title Styling */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }

    /* Header Styling */
    h2, h3 {
        font-weight: 600 !important;
        color: #E0E0E0 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }

    /* Widget Label Styling */
    .stSelectbox label, .stMarkdown p {
        font-size: 1.3rem !important;
        color: #DDDDDD !important;
        font-weight: 500;
    }

    /* Metric Box Styling */
    .metric-card {
        background-color: rgba(0, 0, 0, 0.75);
        border: 1px solid #444;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: bold;
        color: #FFF;
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #AAA;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Background Iframe Styling */
    .stApp {
        background: transparent !important;
    }
    iframe.background-3d {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        border: none;
        margin: 0;
        padding: 0;
        overflow: hidden;
        z-index: -1; 
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 10, 0.9) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- BACKGROUND INJECTION ---
def set_background(html_file):
    with open(html_file, "r") as f:
        html_data = f.read()
    b64_html = base64.b64encode(html_data.encode()).decode()
    st.markdown(
        f'<iframe src="data:text/html;base64,{b64_html}" class="background-3d"></iframe>',
        unsafe_allow_html=True
    )


set_background("animation.html")


# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_data():
    model = joblib.load('jet_engine_model.pkl')
    col_names = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                ['s_{}'.format(i) for i in range(1, 22)]
    test_data = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)

    def prepare_data(df):
        sensor_cols = ['s_{}'.format(i) for i in range(1, 22)]
        rolling = df.groupby('unit_nr')[sensor_cols].rolling(window=5, min_periods=1).mean()
        rolling = rolling.reset_index(level=0, drop=True)
        rolling.columns = [c + '_smooth' for c in sensor_cols]
        df = df.join(rolling)
        return df

    test_data = prepare_data(test_data)
    truth_data = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['True_RUL'])
    return model, test_data, truth_data


model, test_data, truth_data = load_data()

# --- MAIN UI ---
st.title("Predictive Maintenance System")

# --- GLOBAL METRIC (RMSE) ---
rmse_score = 0
try:
    last_rows = test_data.groupby('unit_nr').last().reset_index()
    features_to_drop = ['unit_nr', 'time_cycles']
    X_test = last_rows.drop(columns=features_to_drop)
    y_pred = model.predict(X_test)
    y_true = truth_data['True_RUL']
    rmse_score = math.sqrt(mean_squared_error(y_true, y_pred))
except:
    pass

# Display RMSE in a clean, professional card
st.markdown(f"""
    <div style="background-color: rgba(0,0,0,0.6); padding: 15px; border-left: 5px solid #00d4ff; margin-bottom: 30px;">
        <span style="font-size: 1.5rem; color: #ddd;">SYSTEM ACCURACY (RMSE):</span>
        <span style="font-size: 2rem; color: #fff; font-weight: bold; margin-left: 15px;">{rmse_score:.2f}</span>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("### CONTROL PANEL")
engine_ids = test_data['unit_nr'].unique()
selected_engine = st.sidebar.selectbox("SELECT ENGINE ID", engine_ids)

sensor_desc = {
    's_2': 'LPC Outlet Temp', 's_3': 'HPC Outlet Temp', 's_4': 'LPT Outlet Temp',
    's_7': 'Total Pressure', 's_11': 'Static Pressure', 's_12': 'Fan Speed',
    's_15': 'Bypass Ratio'
}

# --- ENGINE ANALYSIS ---
st.markdown(f"## ENGINE #{selected_engine} STATUS REPORT")

engine_data = test_data[test_data['unit_nr'] == selected_engine]
current_state = engine_data.iloc[-1:]
features = current_state.drop(columns=['unit_nr', 'time_cycles'])
predicted_rul = model.predict(features)[0]

# Health Logic
health_score = (predicted_rul / 150) * 100
health_score = min(max(health_score, 0), 100)

# Determine Status Color
status_text = "HEALTHY"
status_color = "#28a745"  # Green
if health_score < 30:
    status_text = "CRITICAL"
    status_color = "#dc3545"  # Red
elif health_score < 70:
    status_text = "WARNING"
    status_color = "#ffc107"  # Yellow

# --- BIG METRICS DISPLAY ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">REMAINING LIFE</div>
            <div class="metric-value">{int(predicted_rul)}</div>
            <div style="color: #aaa; font-size: 1rem;">CYCLES</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">HEALTH SCORE</div>
            <div class="metric-value">{health_score:.1f}%</div>
            <div style="color: #aaa; font-size: 1rem;">CONDITION</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card" style="border-color: {status_color};">
            <div class="metric-label">CURRENT STATUS</div>
            <div class="metric-value" style="color: {status_color}; font-size: 2.5rem; line-height: 1.6;">{status_text}</div>
            <div style="color: #aaa; font-size: 1rem;">ACTION REQUIRED</div>
        </div>
    """, unsafe_allow_html=True)

# --- VISUALIZATION ---
st.markdown("### SENSOR TELEMETRY")
sensor_options = list(sensor_desc.keys())
selected_sensor = st.selectbox("SELECT SENSOR FEED", sensor_options, format_func=lambda x: f"{x} - {sensor_desc[x]}")

# Professional Chart Styling
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_alpha(0.0)
ax.set_facecolor((0, 0, 0, 0.6))

# Plot Lines
ax.plot(engine_data['time_cycles'], engine_data[selected_sensor], color='#4fc3f7', label='RAW SIGNAL', alpha=0.5,
        linewidth=1)
ax.plot(engine_data['time_cycles'], engine_data[selected_sensor + '_smooth'], color='#ff9800', linewidth=3,
        label='SMOOTHED TREND')

# Large Fonts for Professional Readability
ax.set_title(f"HISTORY: {sensor_desc[selected_sensor]}", color='white', fontsize=18, pad=20)
ax.set_xlabel("FLIGHT CYCLES", color='#cccccc', fontsize=14, labelpad=10)
ax.set_ylabel("SENSOR VALUE", color='#cccccc', fontsize=14, labelpad=10)
ax.tick_params(colors='white', which='both', labelsize=12)
ax.spines['bottom'].set_color('#555555')
ax.spines['left'].set_color('#555555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(facecolor='black', edgecolor='#555555', labelcolor='white', fontsize=12, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.4, color='#555555')

st.pyplot(fig)