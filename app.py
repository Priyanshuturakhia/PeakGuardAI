import datetime as dt
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import altair as alt
import textwrap

# -------------------------------------------------------------------
# 1. PAGE CONFIG & CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="PeakGuard AI",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Global Dark Theme */
    .stApp {
        background: radial-gradient(circle at top, #0f172a 0%, #020617 80%);
    }

    /* Control Panel */
    .control-panel {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #334155;
    }

    /* IMPACT CARDS */
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(30, 41, 59, 0.5);
    }
    .card-green { background-color: rgba(34, 197, 94, 0.15); border-color: #22c55e; }
    .card-red { background-color: rgba(239, 68, 68, 0.15); border-color: #ef4444; }

    .metric-card h3 { font-size: 0.9rem; margin: 0; opacity: 0.8; color: #cbd5e1; }
    .metric-card h1 { font-size: 2.2rem; margin: 5px 0; font-weight: 700; color: #fff; }
    .metric-card p { font-size: 0.8rem; margin: 0; opacity: 0.7; color: #94a3b8; }

    /* === FUTURISTIC AI UI STYLES === */
    .ai-container {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0;
        overflow: hidden;
        margin-top: 0px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    .ai-header {
        padding: 15px 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: 0.5px;
    }
    .ai-header.critical { background: linear-gradient(90deg, #7f1d1d 0%, #450a0a 100%); color: #fecaca; border-bottom: 1px solid #ef4444; }
    .ai-header.safe { background: linear-gradient(90deg, #064e3b 0%, #065f46 100%); color: #d1fae5; border-bottom: 1px solid #34d399; }
    .ai-header.auto { background: linear-gradient(90deg, #1e3a8a 0%, #172554 100%); color: #bfdbfe; border-bottom: 1px solid #3b82f6; }

    .ai-body { padding: 20px; }

    .ai-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 5px;
        margin-top: 15px;
    }
    .ai-label:first-child { margin-top: 0; }

    .ai-text { font-size: 0.95rem; line-height: 1.5; color: #e2e8f0; margin-bottom: 10px; }

    /* Button Override */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# 2. SESSION STATE & LOAD RESOURCES
# -------------------------------------------------------------------
if 'battery_active' not in st.session_state:
    st.session_state.battery_active = False
if 'hvac_active' not in st.session_state:
    st.session_state.hvac_active = False
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False


@st.cache_resource
def load_resources():
    model = lgb.Booster(model_file='lgbm_model.txt')
    feature_names = joblib.load('feature_names.pkl')
    try:
        categorical_features = joblib.load('categorical_features.pkl')
    except:
        categorical_features = []
    meta = pd.read_csv('building_metadata.csv')
    return model, feature_names, categorical_features, meta


def create_donut(val, max_v, label, colors):
    pct = min((val / max_v) * 100, 100) if max_v > 0 else 0
    rem = 100 - pct
    source = pd.DataFrame({"c": [label, ""], "v": [pct, rem]})
    base = alt.Chart(source).encode(theta=alt.Theta("v", stack=True))
    pie = base.mark_arc(innerRadius=55, outerRadius=75).encode(
        color=alt.Color("c", scale=alt.Scale(domain=[label, ""], range=colors), legend=None)
    )
    text = base.mark_text(radius=0, size=18, color="white").encode(text=alt.value(f"{int(pct)}%"))
    return (pie + text).properties(height=180)


model, feature_names, categorical_features, metadata = load_resources()

# -------------------------------------------------------------------
# 3. CONTROL PANEL
# -------------------------------------------------------------------
st.title("🌱 PeakGuard AI")

with st.expander("⚙️ **Simulation Control Panel**", expanded=True):
    # [NEW] MASTER TOGGLE AT THE TOP
    col_main, col_spacer = st.columns([2, 1])
    with col_main:
        st.session_state.auto_mode = st.toggle("Enable Auto-Pilot Mode",
                                               value=st.session_state.auto_mode)

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("BUILDING PROFILE")
        primary_uses = metadata['primary_use'].unique()
        selected_use = st.selectbox("Type", primary_uses, index=0)
        subset = metadata[metadata['primary_use'] == selected_use]
        sq_ft = st.number_input("Area (sq ft)", value=int(subset['square_feet'].mean()), step=5000)
        contract_limit = st.number_input("⚡ Contract Limit (kW)", value=500.0, step=10.0)

    with c2:
        st.caption("ENVIRONMENT & SOLAR")
        solar_capacity = st.number_input("☀️ Solar Capacity (kW)", value=100.0, step=10.0)
        current_temp = st.slider("Outdoor Temp (°C)", -10, 42, 28)
        st.write("")

    with c3:
        st.caption("TIME & HISTORY")
        hour_pick = st.slider("Hour of Day (24h)", 0, 23, 14)
        lag_1 = st.number_input("Load 1hr ago (kW)", value=300.0)
        lag_24 = st.number_input("Load 24hr ago (kW)", value=310.0)

# -------------------------------------------------------------------
# 4. CALCULATION CORE & AUTOMATION LOGIC
# -------------------------------------------------------------------
# Pricing Logic
if 16 <= hour_pick <= 21:  # Peak Hours (4 PM - 9 PM)
    electricity_rate = 24.0
    rate_label = "🔴 PEAK RATE (₹24/kWh)"
    tariff_color = "#ef4444"
elif 13 <= hour_pick <= 16:
    electricity_rate = 18.0
    rate_label = "🟠 HIGH RATE (₹18/kWh)"
    tariff_color = "#f97316"
else:
    electricity_rate = 10.0
    rate_label = "🟢 NORMAL RATE (₹10/kWh)"
    tariff_color = "#22c55e"

# Input Vector
input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = 0
input_data['square_feet'] = sq_ft
input_data['year_built'] = 2005
input_data['floor_count'] = 1
input_data['air_temperature'] = current_temp
input_data['cloud_coverage'] = 2
input_data['dew_temperature'] = current_temp - 5
input_data['month'] = 6
input_data['hour_sin'] = np.sin(2 * np.pi * hour_pick / 24)
input_data['hour_cos'] = np.cos(2 * np.pi * hour_pick / 24)
input_data['day_of_week_sin'] = 0
input_data['day_of_week_cos'] = 1
input_data['meter_reading_lag1'] = lag_1
input_data['meter_reading_lag24'] = lag_24
if f"primary_use_{selected_use}" in input_data.columns:
    input_data[f"primary_use_{selected_use}"] = 1

# Base Prediction
raw_log_pred = model.predict(input_data)[0]
raw_pred = np.expm1(raw_log_pred)
if raw_pred > 200000: raw_pred = raw_log_pred
raw_pred = max(0, raw_pred)

is_daytime = 7 <= hour_pick <= 18
sun_intensity = 0.95
solar_gen = 0.0
if is_daytime:
    efficiency = max(0, 1 - abs(hour_pick - 13) / 6)
    solar_gen = solar_capacity * efficiency * sun_intensity

# --- INTELLIGENT MITIGATION LOGIC ---
mitigation_impact = 0.0
base_load = max(0, raw_pred - solar_gen)
potential_breach = base_load > contract_limit

# Track individual contributions for the report
battery_contrib = 0.0
hvac_contrib = 0.0

# 1. AUTO-PILOT MODE
auto_triggered = False
if st.session_state.auto_mode and potential_breach:
    battery_contrib = 50.0
    hvac_contrib = raw_pred * 0.15
    mitigation_impact = battery_contrib + hvac_contrib
    auto_triggered = True

# 2. MANUAL MODE
else:
    if st.session_state.battery_active:
        battery_contrib = 50.0
        mitigation_impact += battery_contrib
    if st.session_state.hvac_active:
        hvac_contrib = raw_pred * 0.15
        mitigation_impact += hvac_contrib

# Net Load Calculation
net_load = max(0, raw_pred - solar_gen - mitigation_impact)
breach = net_load > contract_limit
excess = net_load - contract_limit
penalty = 25000 if breach else 0

# Financials
saved_co2 = (solar_gen + mitigation_impact) * 0.45
grid_emissions = net_load * 0.45
cost_no_solar = raw_pred * electricity_rate
cost_with_solar = net_load * electricity_rate
money_saved = cost_no_solar - cost_with_solar

# -------------------------------------------------------------------
# 5. DASHBOARD UI
# -------------------------------------------------------------------
st.divider()

# ROW 1: METRICS
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("#### ⚡ Grid Load")
    col = ['#ef4444', '#334155'] if breach else ['#22c55e', '#334155']
    st.altair_chart(create_donut(net_load, contract_limit, "Load", col), use_container_width=True)
    st.caption(f"**{int(net_load)} kW** / {int(contract_limit)} kW")

with c2:
    st.markdown("#### ☀️ Solar Gen")
    st.altair_chart(create_donut(solar_gen, raw_pred, "Solar", ['#facc15', '#334155']), use_container_width=True)
    st.caption(f"**{int(solar_gen)} kW** Generated")

with c3:
    st.markdown("#### 🌱 Avoided CO₂")
    st.markdown(f"""
    <div class="metric-card card-green">
        <h3>Emissions Saved</h3>
        <h1>{saved_co2:.1f} kg</h1>
        <p>vs. standard grid mix</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("#### 🏭 Grid Carbon")
    st.markdown(f"""
    <div class="metric-card card-red">
        <h3>Grid Footprint</h3>
        <h1>{grid_emissions:.1f} kg</h1>
        <p>Dirty energy used</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ROW 2: DETAILED INSIGHTS & AI
col_fin, col_ai = st.columns([1.5, 2])

with col_fin:
    st.subheader("💵 Cost Analysis (Indian Rupees)")
    st.markdown(f"**Current Tariff:** <span style='color:{tariff_color}; font-weight:bold'>{rate_label}</span>",
                unsafe_allow_html=True)

    df = pd.DataFrame({
        'Scenario': ['No AI/Solar', 'With PeakGuard'],
        'Cost': [cost_no_solar, cost_with_solar],
        'Color': ['#ef4444', '#22c55e']
    })
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
        x=alt.X('Scenario', axis=None),
        y='Cost',
        color=alt.Color('Color', scale=None),
        tooltip=['Cost']
    ).properties(height=180)
    text = chart.mark_text(dy=-10, color='white').encode(text=alt.Text('Cost', format='$.0f'))
    st.altair_chart(chart + text, use_container_width=True)

    st.success(f"**💰 Net Saving: ₹{money_saved:,.2f} / hr**")

with col_ai:
    st.subheader("🧠 PeakGuard Intelligence")

    # Determine UI State
    is_mitigated = st.session_state.battery_active or st.session_state.hvac_active

    if auto_triggered:
        # AUTO-PILOT STATE
        header_class = "auto"
        icon = "🤖"
        title = "AUTO-PILOT ENGAGED"

        # [NEW] DETAILED BREAKDOWN
        diag_text = f"""
        <b>Threat Neutralized.</b> AI instantly deployed <b style='color:#3b82f6'>{int(mitigation_impact)} kW</b> countermeasures.<br><br>
        <b>OPTIMIZATION LOG:</b><br>
        • 🔋 Battery Dispatch: <b>{int(battery_contrib)} kW</b><br>
        • ❄️ HVAC Shift (1.5°C): <b>{int(hvac_contrib)} kW</b>
        """
        root_cause = "Autonomous System response to predicted grid breach."

    elif breach:
        # DANGER STATE
        header_class = "critical"
        icon = "🚨"
        title = "CRITICAL BREACH DETECTED"
        diag_text = f"System is <b style='color:#ef4444'>+{int(excess)} kW</b> over limit."
        root_cause = "Peak Tariff Hours coinciding with high AC load."

    elif is_mitigated:
        # MANUAL FIX STATE
        header_class = "safe"
        icon = "🛡️"
        title = "MITIGATION ACTIVE"
        diag_text = f"Manual Actions reduced load by <b style='color:#22c55e'>{int(mitigation_impact)} kW</b>."
        root_cause = "Operator intervention successful."

    else:
        # SAFE STATE
        header_class = "safe"
        icon = "✅"
        title = "SYSTEM OPTIMIZED"
        diag_text = f"Load is <b style='color:#22c55e'>{int(contract_limit - net_load)} kW</b> below limit."
        root_cause = "Passive solar integration effective."

    # --- CARD HEADER (HTML) ---
    html_top = ""
    html_top += f'<div class="ai-container">'
    html_top += f'  <div class="ai-header {header_class}">'
    html_top += f'    <span style="font-size:1.5rem">{icon}</span> {title}'
    html_top += f'  </div>'
    html_top += f'  <div class="ai-body">'
    html_top += f'    <div class="ai-label">DIAGNOSIS</div>'
    html_top += f'    <div class="ai-text">{diag_text}</div>'
    html_top += f'    <div class="ai-label">ROOT CAUSE</div>'
    html_top += f'    <div class="ai-text" style="color:#cbd5e1; font-style:italic;">"{root_cause}"</div>'
    html_top += f'    <div class="ai-label">RECOMMENDED ACTIONS</div>'

    st.markdown(html_top, unsafe_allow_html=True)

    # --- INTERACTIVE CONTROLS ---
    if auto_triggered:
        # Auto Mode Display
        st.info("⚡ Autonomous Protocol Executed. System Secure.")

    elif breach:
        # Manual Breach - Show Buttons
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔋 Dispatch Battery"):
                st.session_state.battery_active = True
                st.rerun()
        with c_btn2:
            if st.button("❄️ Optimize HVAC"):
                st.session_state.hvac_active = True
                st.rerun()
        st.caption(f"Risk: ₹{penalty:,} penalty accruing now.")

    elif is_mitigated:
        # Reset Button
        st.success("✅ Intervention Successful")
        if st.button("🔄 Reset System"):
            st.session_state.battery_active = False
            st.session_state.hvac_active = False
            st.rerun()

    else:
        # Safe State
        if solar_gen > 50:
            safe_msg = '<div class="action-item opportunity"><b>❄️ PRE-COOL:</b> Use solar to supercool water loops.</div>'
        else:
            safe_msg = '<div class="action-item opportunity"><b>🔋 CHARGE:</b> Grid load is low; recharge main battery.</div>'
        st.markdown(safe_msg, unsafe_allow_html=True)

    # --- CARD FOOTER ---
    st.markdown("</div></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 6. REPORTING (SIDEBAR)
# -------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("📋 Shift Reporting")

    report_content = textwrap.dedent(f"""
        PEAKGUARD AI - INCIDENT REPORT
        ----------------------------------
        Date: {dt.date.today()}
        Time Block: {hour_pick}:00 - {hour_pick + 1}:00
        Building Type: {selected_use}
        Auto-Pilot: {'ENABLED' if st.session_state.auto_mode else 'DISABLED'}

        STATUS: {'CRITICAL BREACH' if breach else 'OPTIMIZED'}
        ----------------------------------
        Contract Limit: {contract_limit} kW
        Actual Net Load: {net_load:.2f} kW

        FINANCIALS (Rate: ₹{electricity_rate}/kWh)
        ----------------------------------
        Net Savings:     ₹{money_saved:.2f}

        ACTIONS:
        [x] Solar Integration
        [{'x' if auto_triggered else ' '}] Auto-Pilot Mitigation
        [{'x' if st.session_state.battery_active else ' '}] Manual Battery
    """).strip()

    st.download_button(
        label="📄 Download Official Report",
        data=report_content,
        file_name=f"PeakGuard_Report_{hour_pick}00.txt",
        mime="text/plain"
    )
    st.caption("Generate compliant PDF/Text logs.")