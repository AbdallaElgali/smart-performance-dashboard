import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import pickle
import io
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

# ==========================================
# 🎨 0. UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Athlete AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust CSS for Metric Cards only (Less fragile)
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00FF99;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .metric-card h3 {
        margin: 0;
        padding: 0;
        font-size: 18px;
        color: #FAFAFA;
    }
    .metric-card h2 {
        margin: 5px 0 0 0;
        font-size: 28px;
        font-weight: bold;
        color: #00FF99;
    }
    /* Force Plotly chart background to match */
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔄 1. DATA PROCESSING ENGINE
# ==========================================
@st.cache_data
def process_excel_file(uploaded_file):
    try:
        raw_df = pd.read_excel(uploaded_file, header=None)

        # Find Headers dynamically
        name_row_idx = raw_df[raw_df.apply(lambda row: row.astype(str).str.contains('ARNAS', case=False).any(), axis=1)].index[0]
        metric_row_idx = name_row_idx + 1
        data_start_idx = metric_row_idx + 1

        player_headers = raw_df.iloc[name_row_idx].ffill()
        metric_headers = raw_df.iloc[metric_row_idx]

        col_map = {}
        def clean_name(n): return str(n).replace("\n", " ").strip()
        
        standard_metrics = {
            "RPEm": "RPEm", "RPEt": "RPEt", "load RPE": "LoadRPE", "LoadRPE": "LoadRPE",
            "load FC": "LoadFC", "A-C RPE": "AC_Ratio", "A RPE": "Acute_Load", "C RPE": "Chronic_Load"
        }
        player_zone_tracker = {}

        for col_idx in range(len(player_headers)):
            p_name = str(player_headers[col_idx]).strip()
            m_name = clean_name(metric_headers[col_idx])

            if p_name in ["nan", "GRÁFICOS", "FECHA", "CARGA", "GENERAL", "0", "0.0", ""]: continue

            final_metric = None
            if m_name in standard_metrics:
                final_metric = standard_metrics[m_name]
            elif m_name in ["Z1", "Z2", "Z3", "Z4", "Z5"]:
                if p_name not in player_zone_tracker:
                    player_zone_tracker[p_name] = {z: 0 for z in ["Z1", "Z2", "Z3", "Z4", "Z5"]}
                current_count = player_zone_tracker[p_name][m_name]
                player_zone_tracker[p_name][m_name] += 1
                final_metric = f"{m_name}m" if current_count == 0 else f"{m_name}t"
            
            if final_metric: col_map[col_idx] = (p_name, final_metric)

        extracted_rows = []
        for i in range(data_start_idx, len(raw_df)):
            row = raw_df.iloc[i]
            val_date = row[2]
            if pd.isna(val_date): continue

            final_date = None
            try:
                if isinstance(val_date, datetime): dt_temp = val_date
                elif isinstance(val_date, str) and "-" in val_date:
                    dt_temp = datetime.strptime(val_date.strip(), "%d-%b")
                else: continue

                if dt_temp.month >= 8: final_date = dt_temp.replace(year=2023)
                else: final_date = dt_temp.replace(year=2024)
            except: continue

            if final_date is None: continue

            for col_idx, (player, metric) in col_map.items():
                val = row[col_idx]
                try: val_float = float(str(val).replace(',', ''))
                except: val_float = 0.0
                extracted_rows.append({"Date": final_date, "Player": player, "Metric": metric, "Value": val_float})

        long_df = pd.DataFrame(extracted_rows)
        df_final = long_df.pivot_table(index=['Date', 'Player'], columns='Metric', values='Value', aggfunc='first').reset_index()
        df_final = df_final.fillna(0).sort_values(by=['Date', 'Player'])
        return df_final

    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 🧠 2. ML TRAINING ENGINE
# ==========================================
def train_models(df):
    df_ml = df.copy()
    df_ml['Date'] = pd.to_datetime(df_ml['Date'])
    df_ml = df_ml.sort_values(['Player', 'Date'])

    zones = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    for z in zones:
        m, t = f'{z}m', f'{z}t'
        df_ml[f'{z}_Total'] = (df_ml[m] if m in df_ml else 0) + (df_ml[t] if t in df_ml else 0)

    df_ml['DayOfWeek'] = df_ml['Date'].dt.dayofweek
    df_ml['Num_Sessions'] = df_ml['RPEt'].apply(lambda x: 2 if x > 0 else 1) if 'RPEt' in df_ml else 1
    df_ml['Roll_7'] = df_ml.groupby('Player')['LoadRPE'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df_ml['Roll_28'] = df_ml.groupby('Player')['LoadRPE'].transform(lambda x: x.rolling(28, min_periods=1).mean())
    df_ml['Load_Yesterday'] = df_ml.groupby('Player')['LoadRPE'].shift(1)
    df_ml['Load_2DaysAgo'] = df_ml.groupby('Player')['LoadRPE'].shift(2)

    df_ml['Target_Load'] = df_ml.groupby('Player')['LoadRPE'].shift(-1)
    future_ac = ((df_ml['Roll_7']*6 + df_ml['Target_Load'])/7) / (((df_ml['Roll_28']*27 + df_ml['Target_Load'])/28) + 1)
    df_ml['Target_Risk'] = future_ac.apply(lambda x: 0 if x < 0.8 else (1 if x <= 1.3 else 2))
    df_ml['Target_Perf'] = df_ml['Target_Load'].apply(lambda x: 0 if x < 300 else (1 if x < 600 else 2))

    df_ready = df_ml.dropna(subset=['Target_Load', 'Load_Yesterday']).query("LoadRPE > 10")
    features = ['DayOfWeek', 'Num_Sessions', 'Load_Yesterday', 'Load_2DaysAgo', 'Roll_7', 'Roll_28'] + [f'{z}_Total' for z in zones]
    
    X = df_ready[features]
    y_load = df_ready['Target_Load']
    y_risk = df_ready['Target_Risk']
    y_perf = df_ready['Target_Perf']

    X_train, X_test, idx_train, idx_test = train_test_split(X.index, X.index, test_size=0.3, random_state=42)
    X_tr, X_te = X.loc[X_train], X.loc[X_test]

    # Using basic XGB params for speed and stability
    reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.1)
    reg.fit(X_tr, y_load.loc[idx_train])
    
    risk = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100, max_depth=4, learning_rate=0.1)
    risk.fit(X_tr, y_risk.loc[idx_train])
    
    perf = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100, max_depth=4, learning_rate=0.1)
    perf.fit(X_tr, y_perf.loc[idx_train])

    scores = {
        'mae': mean_absolute_error(y_load.loc[idx_test], reg.predict(X_te)),
        'acc_risk': accuracy_score(y_risk.loc[idx_test], risk.predict(X_te)),
        'acc_perf': accuracy_score(y_perf.loc[idx_test], perf.predict(X_te))
    }
    return reg, risk, perf, scores, X_te, y_load.loc[idx_test], y_risk.loc[idx_test], y_perf.loc[idx_test]

# ==========================================
# 🖥️ 3. MAIN APP UI
# ==========================================

st.sidebar.image("https://img.icons8.com/fluency/96/running.png", width=80)
st.sidebar.title("Athlete AI")
st.sidebar.markdown("Data-Driven Performance")
uploaded_file = st.sidebar.file_uploader("📂 Upload Data (.xlsx)", type=['xlsx'])

if 'models' not in st.session_state:
    st.session_state['models'] = None

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "📂 Data", "📊 Analytics", "🧠 Models", "🔮 Simulator"])

# --- TAB 1: HOME ---
with tab1:
    st.markdown("## ⚡ High-Performance Athlete Analytics")
    st.markdown("""
    Turn your raw Excel data into actionable intelligence using AI.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Pipeline Status</h3>
            <h2>Ready</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Models Available</h3>
            <h2>3 (Reg/Class)</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Upload Limit</h3>
            <h2>1 GB</h2>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: DATA ---
with tab2:
    if uploaded_file:
        with st.spinner("ETL Processing in progress..."):
            result = process_excel_file(uploaded_file)
        
        if isinstance(result, str):
            st.error(result)
        else:
            st.session_state['df'] = result
            st.success(f"✅ Successfully processed {len(result)} records.")
            st.dataframe(result.head(100), height=400, use_container_width=True)
            
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Clean CSV", csv, "clean_data.csv", "text/csv")
    else:
        st.info("👈 Upload your Excel file in the sidebar to begin.")

# --- TAB 3: ANALYTICS ---
with tab3:
    if 'df' in st.session_state:
        df = st.session_state['df']
        c1, c2 = st.columns([1, 4])
        with c1:
            player = st.selectbox("Player", df['Player'].unique())
            metric = st.selectbox("Metric", ['LoadRPE', 'AC_Ratio', 'LoadFC', 'RPEm', 'RPEt'])
        with c2:
            subset = df[df['Player'] == player]
            fig = px.line(subset, x='Date', y=metric, title=f"{player} - {metric}", template="plotly_dark")
            if metric == 'AC_Ratio':
                fig.add_hrect(y0=1.3, y1=2.0, fillcolor="red", opacity=0.1, annotation_text="High Risk")
                fig.add_hrect(y0=0.8, y1=1.3, fillcolor="green", opacity=0.1, annotation_text="Optimal")
            st.plotly_chart(fig, use_container_width=True)
            
            # Zones
            z_cols = [c for c in df.columns if 'Z' in c and ('m' in c or 't' in c)]
            if z_cols:
                z_sum = subset[z_cols].sum().reset_index()
                z_sum.columns = ['Zone', 'Minutes']
                fig_z = px.bar(z_sum, x='Zone', y='Minutes', color='Minutes', template="plotly_dark", title="Zone Distribution")
                st.plotly_chart(fig_z, use_container_width=True)
    else:
        st.warning("No data loaded.")

# --- TAB 4: MODELS ---
with tab4:
    if 'df' in st.session_state:
        if st.button("🚀 Train AI Models", type="primary"):
            with st.spinner("Training XGBoost Ensemble..."):
                reg, risk, perf, scores, X_te, y_reg, y_risk, y_perf = train_models(st.session_state['df'])
                st.session_state['models'] = {'reg': reg, 'risk': risk, 'perf': perf}
                
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-card'><h3>Forecast Error (MAE)</h3><h2>{scores['mae']:.1f}</h2></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><h3>Risk Accuracy</h3><h2>{scores['acc_risk']*100:.1f}%</h2></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><h3>Intensity Accuracy</h3><h2>{scores['acc_perf']*100:.1f}%</h2></div>", unsafe_allow_html=True)
                
                # Viz
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    # Risk CM
                    preds = risk.predict(X_te)
                    cm = confusion_matrix(y_risk, preds)
                    fig_cm = px.imshow(cm, text_auto=True, x=['Low', 'Med', 'High'], y=['Low', 'Med', 'High'],
                                     labels=dict(x="Pred", y="Actual"), title="Risk Confusion Matrix", template="plotly_dark")
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col_v2:
                    # Reg Scatter
                    preds_r = reg.predict(X_te)
                    fig_sc = px.scatter(x=y_reg, y=preds_r, title="Forecast vs Actual", template="plotly_dark",
                                      labels={'x':'Actual', 'y':'Predicted'})
                    fig_sc.add_shape(type="line", x0=0, y0=0, x1=1000, y1=1000, line=dict(color="red", dash="dash"))
                    st.plotly_chart(fig_sc, use_container_width=True)
                    
                # Pickle
                model_pack = {'reg': reg, 'risk': risk, 'perf': perf}
                b = io.BytesIO()
                pickle.dump(model_pack, b)
                st.download_button("💾 Save Models (.pkl)", b, "athlete_ai.pkl")
    else:
        st.warning("Load data first.")

# --- TAB 5: SIMULATOR ---
with tab5:
    st.markdown("### 🔮 The Coach's Cockpit")
    if st.session_state['models']:
        c_in, c_out = st.columns([1, 2])
        with c_in:
            st.markdown("#### Scenario Settings")
            day = st.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            sess = st.radio("Sessions", [1, 2], horizontal=True)
            load_y = st.slider("Yesterday Load", 0, 1000, 450)
            fit = st.slider("Fitness (Chronic)", 200, 1000, 500)
            fat = st.slider("Fatigue (1-10)", 1, 10, 5)
            
        with c_out:
            if st.button("Run Simulation", type="primary"):
                day_map = {"Mon":0, "Tue":1, "Wed":2, "Thu":3, "Fri":4, "Sat":5, "Sun":6}
                roll_7 = fit * (1 + (fat-5)/10)
                z_avg = load_y/5
                
                input_row = pd.DataFrame([{
                    'DayOfWeek': day_map[day], 'Num_Sessions': sess, 'Load_Yesterday': load_y,
                    'Load_2DaysAgo': load_y*0.8, 'Roll_7': roll_7, 'Roll_28': fit,
                    'Z1_Total': z_avg, 'Z2_Total': z_avg, 'Z3_Total': z_avg, 'Z4_Total': z_avg, 'Z5_Total': z_avg
                }])
                
                mods = st.session_state['models']
                p_l = mods['reg'].predict(input_row)[0]
                p_r = mods['risk'].predict(input_row)[0]
                p_p = mods['perf'].predict(input_row)[0]
                
                r_lbl = {0: "LOW RISK", 1: "OPTIMAL", 2: "HIGH RISK"}
                r_col = {0: "#00FF00", 1: "#00AAFF", 2: "#FF0000"}
                p_lbl = {0: "Recovery", 1: "Maintenance", 2: "Hard"}
                
                st.markdown(f"""
                <div style="display: flex; gap: 20px;">
                    <div class="metric-card" style="flex: 1;">
                        <h3>Predicted Load</h3>
                        <h2>{p_l:.0f} AU</h2>
                    </div>
                    <div class="metric-card" style="flex: 1; border-left: 5px solid {r_col[p_r]}">
                        <h3>Risk Status</h3>
                        <h2 style="color:{r_col[p_r]}">{r_lbl[p_r]}</h2>
                    </div>
                     <div class="metric-card" style="flex: 1;">
                        <h3>Intensity</h3>
                        <h2>{p_lbl[p_p]}</h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Train models first.")