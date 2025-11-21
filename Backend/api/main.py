from flask import Flask, jsonify, request
from flask_cors import CORS
from dbconn import Database
from datetime import datetime, timedelta
import statistics
from datetime import date
import pickle
import pandas as pd
import numpy as np

# ================= MODEL LOADING =================
# Load models once at startup to save performance
try:
    with open('models/model_forecast_xgb.pkl', 'rb') as f:
        load_model = pickle.load(f)
    with open('models/model_risk_xgb.pkl', 'rb') as f:
        risk_model = pickle.load(f)
    print("✅ ML Models loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    load_model = None
    risk_model = None

# ================= CONFIGURATION =================
app = Flask(__name__)
CORS(app)  # Enable CORS for Frontend communication

# Database Config (Update with your local credentials)
DB_CONFIG = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_bpru5JTgn3iO',
    'host': 'ep-steep-poetry-a45xescd-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

# Initialize DB Class
db = Database(DB_CONFIG)


# ================= ENDPOINTS =================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple check to see if API is running."""
    return jsonify({"status": "ok", "message": "Basketball Analytics API is live"}), 200


# --------------------------------- ----------------
# 1. PLAYER MANAGEMENT
# -------------------------------------------------
@app.route('/api/players', methods=['GET'])
def get_all_players():
    """Fetch list of all players for roster views."""
    query = "SELECT player_id, name, jersey_number, weight_kg, age, height_m, position, photo_url, source_url FROM players ORDER BY player_id;"
    players = db.execute(query, fetch_all=True)
    if players:
        return jsonify(players), 200
    return jsonify({"error": "No players found"}), 404



# -------------------------------------------------
# HELPERS
#--------------------------------------------------
def acwr_readiness(ac):
    if 0.8 <= ac <= 1.3:
        return 100
    elif 0.6 <= ac < 0.8:
        return 70
    elif 1.3 < ac <= 1.5:
        return 70
    elif 0.4 <= ac < 0.6:
        return 50
    elif 1.5 < ac <= 2.0:
        return 50
    else:
        return 0

def monotony_readiness(monotony):
    if monotony < 1.5:
        return 100
    elif 1.5 <= monotony <= 2.0:
        return 70
    else:
        return 0
def rpe_readiness(recent_rpes):
    if not recent_rpes:
        return 100
    avg_rpe = sum(recent_rpes) / len(recent_rpes)
    if all(4 <= r <= 6 for r in recent_rpes):
        return 70
    elif avg_rpe < 4:
        return 50
    else:
        return 100

# -------------------------------------------------
# 3. PLAYER DEEP DIVE (Charts)
# -------------------------------------------------
@app.route('/api/player/<int:player_id>/trends', methods=['GET'])
def get_player_trends(player_id):
    """
    Analyze last 30 days of Load/RPE to predict Illness, Injury, Freshness, and Readiness.
    """
    # --- Fetch Data ---
    query = """
        SELECT 
            session_id,
            activity_date,
            load_rpe as external_load,
            rpe_score,
            ac_ratio
        FROM sessions
        WHERE player_id = %s
        ORDER BY activity_date ASC
        LIMIT 30;
    """
    data = db.execute(query, (player_id,), fetch_all=True)

    # Fill missing days with rest-day entries (load=0, rpe=0)
    filled = []
    prev_date = None
    for row in data:
        current_date = row["activity_date"]
        if prev_date is not None:
            delta = (current_date - prev_date).days
            if delta > 1:
                for i in range(1, delta):
                    missing_date = prev_date + timedelta(days=i)
                    filled.append({
                        "session_id": None,
                        "activity_date": missing_date,
                        "external_load": 0,
                        "rpe_score": 0,
                        "ac_ratio": None
                    })
        filled.append(row)
        prev_date = current_date
    data = filled

    if not data:
        return jsonify([]), 200

    # --- Prepare Metrics ---
    loads = [d['external_load'] for d in data if d['external_load'] is not None]
    latest = data[-1]
    current_ac = latest.get('ac_ratio') or 0

    # --- INSIGHT ENGINE ---
    insights = []

    # Insight 1: Illness Risk (Training Monotony)
    if len(loads) >= 7:
        last_7_loads = loads[-7:]
        avg_load = sum(last_7_loads) / 7
        variance = sum((x - avg_load) ** 2 for x in last_7_loads) / 7
        std_dev = variance ** 0.5
        monotony = avg_load / std_dev if std_dev > 0 else 0

        if monotony > 2.0:
            insights.append({
                "type": "CRITICAL",
                "title": "Illness Risk Elevated",
                "score": 9,
                "message": f"Training Monotony is {monotony:.1f} (High). Lack of rest days detected.",
                "action": "Mandatory Rest Day or Active Recovery tomorrow to reset variation."
            })
        elif monotony > 1.5:
            insights.append({
                "type": "WARNING",
                "title": "Staleness Risk",
                "score": 6,
                "message": "Training variation is low. Player may feel 'stale' or mentally fatigued.",
                "action": "Vary the intensity of the next drill (High/Low)."
            })

    # Insight 2: Injury Risk (ACWR Sweet Spot)
    if current_ac > 1.5:
        insights.append({
            "type": "CRITICAL",
            "title": "High Injury Risk",
            "score": 10,
            "message": f"Workload spiked 50% above baseline (AC {current_ac:.2f}). Soft tissue risk is critical.",
            "action": "Cap Volume at -40% for next 3 days."
        })
    elif current_ac < 0.8 and current_ac > 0.1:
        insights.append({
            "type": "INFO",
            "title": "Detraining Status",
            "score": 5,
            "message": f"AC Ratio is {current_ac:.2f}. Player is losing fitness or tapering.",
            "action": "If not tapering for a game, increase volume."
        })
    elif 0.8 <= current_ac <= 1.3:
        insights.append({
            "type": "POSITIVE",
            "title": "Optimal Loading",
            "score": 3,
            "message": "Player is in the 'Sweet Spot' for adaptation and safety.",
            "action": "Maintain current progression."
        })

    # Insight 3: Grey Zone / Intensity Check
    if len(data) >= 5:
        recent_rpes = [d['rpe_score'] for d in data[-5:] if d['rpe_score'] is not None]
        if recent_rpes:
            avg_rpe = sum(recent_rpes) / len(recent_rpes)
            if all(4 <= r <= 6 for r in recent_rpes):
                insights.append({
                    "type": "WARNING",
                    "title": "Grey Zone Training",
                    "score": 7,
                    "message": "Last 5 sessions were all Moderate Intensity (RPE 4-6). No high stimulus detected.",
                    "action": "Prescribe either a Sprint Session (RPE 8+) or Deep Recovery (RPE <3)."
                })

    insights.sort(key=lambda x: x['score'], reverse=True)

    # --- READINESS SCORE CALCULATION ---
    acwr_subscore = acwr_readiness(current_ac)
    monotony_subscore = monotony_readiness(monotony if len(loads) >= 7 else 0)
    rpe_subscore = rpe_readiness(recent_rpes if len(data) >= 5 else [])

    readiness_score = round(
        0.4 * acwr_subscore +
        0.3 * monotony_subscore +
        0.3 * rpe_subscore
    )

    if readiness_score >= 85:
        readiness_status = "Fully Ready"
    elif readiness_score >= 70:
        readiness_status = "Moderately Ready"
    elif readiness_score >= 50:
        readiness_status = "Low Readiness – caution advised"
    else:
        readiness_status = "High Risk – Rest or Recovery Required"

    response = {
        "chart_data": data,
        "insights": insights,
        "readiness": {
            "score": readiness_score,
            "status": readiness_status
        }
    }

    return jsonify(response), 200


# -------------------------------------------------
# 5. TEAM ANALYSIS (Macro View) - HISTORICAL DATA FIX
# -------------------------------------------------
@app.route('/api/analysis/team', methods=['GET'])
def get_team_analysis():
    """
    Aggregates data for the last 7 and 28 days RELATIVE TO THE LATEST DATA POINT.
    This allows analysis of historical datasets where CURRENT_DATE would return nothing.
    """
    query = """
        WITH LatestData AS (
            -- 1. Find the most recent date in the database
            SELECT MAX(activity_date) as anchor_date FROM sessions
        )
        SELECT 
            p.player_id, 
            p.name, 
            p.player_position as position,
            -- Count sessions in the last 28 days relative to anchor
            COUNT(CASE WHEN s.activity_date > (ld.anchor_date - INTERVAL '28 days') THEN 1 END) as session_count,

            -- Acute Load: Sum of load in last 7 days relative to anchor
            SUM(CASE 
                WHEN s.activity_date > (ld.anchor_date - INTERVAL '7 days') 
                THEN s.load_rpe ELSE 0 
            END) as acute_load,

            -- Chronic Load: Sum of load in last 28 days relative to anchor
            SUM(CASE 
                WHEN s.activity_date > (ld.anchor_date - INTERVAL '28 days') 
                THEN s.load_rpe ELSE 0 
            END) as chronic_sum,

            -- Pass the anchor date back so frontend knows when this data is from
            ld.anchor_date

        FROM players p
        CROSS JOIN LatestData ld
        LEFT JOIN sessions s ON p.player_id = s.player_id 
        -- Optimization: Only join relevant recent rows to keep query fast
        AND s.activity_date > (ld.anchor_date - INTERVAL '28 days')

        GROUP BY p.player_id, p.name, p.player_position, ld.anchor_date;
    """

    data = db.execute(query, fetch_all=True)

    if not data:
        return jsonify({"error": "No data found", "roster": [], "team_avg_load": 0}), 200

    processed_roster = []

    # Calculate Team Averages for Context
    all_acute_loads = [float(d['acute_load'] or 0) for d in data]
    team_avg_acute = statistics.mean(all_acute_loads) if all_acute_loads else 0

    for p in data:
        acute = float(p['acute_load'] or 0)
        chronic_total = float(p['chronic_sum'] or 0)

        # Chronic Avg Load = Total Load / 28 days
        chronic_avg = chronic_total / 28 if chronic_total > 0 else 1

        # ACWR Formula
        acwr = round(acute / (chronic_avg * 7), 2) if chronic_avg > 1 else 0

        # --- COACH INSIGHTS ---
        status = "Optimal"
        flag = "green"

        if acwr > 1.5:
            status = "High Injury Risk (Spike)"
            flag = "red"
        elif acwr < 0.8:
            status = "Detraining / Tapering"
            flag = "yellow"
        elif acute > (team_avg_acute * 1.5):
            status = "Workhorse (High Volume)"
            flag = "orange"

        processed_roster.append({
            "name": p['name'],
            "position": p['position'],
            "acute_load": acute,
            "acwr": acwr,
            "status": status,
            "flag": flag,
            "sessions_last_28": p['session_count']
        })

    # Sort by Acute Load (Highest first)
    processed_roster.sort(key=lambda x: x['acute_load'], reverse=True)

    return jsonify({
        "team_avg_load": round(team_avg_acute, 1),
        "last_data_date": data[0]['anchor_date'] if data else None,
        "roster": processed_roster
    }), 200


@app.route('/api/analysis/player/<int:player_id>', methods=['GET'])
def analyze_player_history(player_id):
    """
    Fetches 90 days of history, fills gaps, and calculates a
    0-100 READINESS SCORE based on ACWR and Monotony.
    """
    # 1. Get Raw Data
    # Using 'duration_minutes' as corrected
    # Note: We use date math instead of LIMIT to ensure we get the *recent* 90 days
    query = """
        SELECT activity_date, load_rpe, rpe_score, duration_min
        FROM sessions
        WHERE player_id = %s
        ORDER BY activity_date ASC
        LIMIT 90;
    """
    raw_data = db.execute(query, (player_id,), fetch_all=True)

    # Handle Empty Data Gracefully
    if not raw_data:
        return jsonify({
            "player_id": player_id,
            "summary": {
                "current_status": "No Data",
                "action_required": "Log training sessions to generate insights.",
                "readiness_score": 0
            },
            "history": []
        }), 200

    # 2. APPLY GAP FILLING LOGIC
    filled = []
    prev_date = None

    for row in raw_data:
        current_date = row["activity_date"]

        # Ensure date object
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, '%Y-%m-%d').date()

        if prev_date is not None:
            delta = (current_date - prev_date).days
            if delta > 1:
                for i in range(1, delta):
                    missing_date = prev_date + timedelta(days=i)
                    filled.append({
                        "activity_date": missing_date,
                        "load_rpe": 0,
                        "rpe_score": 0,
                        "is_computed_rest": True
                    })

        row['is_computed_rest'] = False
        if isinstance(row['activity_date'], str):
            row['activity_date'] = datetime.strptime(row['activity_date'], '%Y-%m-%d').date()

        filled.append(row)
        prev_date = current_date

    data = filled

    # 3. CALCULATE METRICS & READINESS SCORE
    analyzed_history = []

    for i in range(len(data)):
        day = data[i]

        # Acute (7 days) & Chronic (28 days) Windows
        start_acute = max(0, i - 6)
        acute_window = [d['load_rpe'] for d in data[start_acute: i + 1]]

        start_chronic = max(0, i - 27)
        chronic_window = [d['load_rpe'] for d in data[start_chronic: i + 1]]

        # Averages
        acute_load = sum(acute_window) / len(acute_window) if acute_window else 0
        chronic_load = sum(chronic_window) / len(chronic_window) if chronic_window else 0

        # ACWR
        acwr = round(acute_load / chronic_load, 2) if chronic_load > 10 else 0

        # Monotony
        if len(acute_window) > 1:
            stdev = statistics.stdev(acute_window)
            monotony = round(statistics.mean(acute_window) / stdev, 2) if stdev > 0 else 0
        else:
            monotony = 0

        # --- READINESS SCORE CALCULATION (0-100) ---
        score = 100

        # A. Injury Risk Penalty (ACWR)
        if acwr > 1.5:
            score -= 40  # Danger Zone
        elif acwr > 1.3:
            score -= 20  # High Risk
        elif acwr < 0.8:
            score -= 10  # Undertrained/Detraining

        # B. Burnout Penalty (Monotony)
        if monotony > 2.5:
            score -= 30
        elif monotony > 1.5:
            score -= 15

        # C. Fatigue Penalty (Acute > Chronic)
        # If they are working much harder this week than usual
        if acute_load > (chronic_load * 1.2):
            score -= 10

        # Clamp score between 0 and 100
        readiness_score = max(0, min(100, score))

        analyzed_history.append({
            "date": day['activity_date'].strftime('%Y-%m-%d'),
            "daily_load": day['load_rpe'],
            "acute_load": round(acute_load, 0),
            "chronic_load": round(chronic_load, 0),
            "acwr": acwr,
            "monotony": monotony,
            "readiness_score": readiness_score,  # Replaces freshness_index
            "is_rest": day.get('is_computed_rest', False)
        })

    # 4. GENERATE FINAL COACH SUMMARY
    latest = analyzed_history[-1]
    current_readiness = latest['readiness_score']

    summary = {
        "readiness_score": current_readiness,
        "current_status": "Optimal",
        "action_required": "Maintain current load."
    }

    # Logic for textual summary based on the Score + Metrics
    if current_readiness < 60:
        summary['current_status'] = "Low Readiness (High Risk)"
        if latest['acwr'] > 1.5:
            summary['action_required'] = "ACWR Spike detected. Cut volume by 40%."
        elif latest['monotony'] > 2.0:
            summary['action_required'] = "Monotony is high. Schedule a Rest Day."
    elif current_readiness < 80:
        summary['current_status'] = "Moderately Fatigued"
        summary['action_required'] = "Monitor intensity during next session."
    else:
        summary['current_status'] = "Peak Condition"
        summary['action_required'] = "Player is ready to perform."

    return jsonify({
        "player_id": player_id,
        "summary": summary,
        "history": analyzed_history
    }), 200


@app.route('/api/predict-session', methods=['POST'])
def predict_session():
    """
    Inputs: { "player_id": 1, "rpe": 7, "duration": 90 }
    1. Fetches history from DB.
    2. Fills date gaps with 0s.
    3. Calculates Rolling Features.
    4. Runs Models (Load + Risk).
    5. Calculates Weighted Risk % based on AI Probability + Mechanical Load + Projected ACWR.
    """
    if not load_model or not risk_model:
        return jsonify({"error": "Models not loaded on server"}), 503

    data = request.json
    player_id = data.get('player_id')
    target_rpe = float(data.get('rpe', 5))
    target_duration = float(data.get('duration', 60))

    # 1. Fetch History
    query = """
        SELECT activity_date, load_rpe 
        FROM sessions 
        WHERE player_id = %s 
        ORDER BY activity_date ASC;
    """
    raw_data = db.execute(query, (player_id,), fetch_all=True)

    if not raw_data:
        return jsonify({"error": "Not enough history to predict"}), 400

    # 2. Process Data (Pandas)
    df = pd.DataFrame(raw_data)
    df['activity_date'] = pd.to_datetime(df['activity_date'])
    df.set_index('activity_date', inplace=True)

    # Fill Gaps
    df = df.resample('D').sum().fillna(0)

    # Rolling Metrics (Current State)
    df['rolling_7'] = df['load_rpe'].rolling(window=7, min_periods=1).mean()
    df['rolling_28'] = df['load_rpe'].rolling(window=28, min_periods=1).mean()
    df['acwr'] = np.where(df['rolling_28'] > 0, df['rolling_7'] / df['rolling_28'], 0)

    # 3. Extract Features
    last_entry = df.iloc[-1]
    prev_entry = df.iloc[-2] if len(df) > 1 else last_entry

    last_date = df.index[-1]
    simulated_date = last_date + pd.Timedelta(days=1)
    day_of_week = simulated_date.dayofweek

    features = np.array([[
        day_of_week,
        last_entry['load_rpe'],
        prev_entry['load_rpe'],
        last_entry['rolling_7'],
        last_entry['rolling_28'],
        last_entry['acwr'],
        target_rpe
    ]])

    # 4. Run Predictions & Calculate Logic
    try:
        # --- A. Hybrid Load Calculation (The fix for "weird" model outputs) ---
        # 1. Get Pure Model Prediction
        raw_pred_load = float(load_model.predict(features)[0])

        # 2. Calculate Heuristic Load (Physics-based sanity check)
        # Assumption: sRPE (Dur * RPE) correlates with Mech Load.
        # A multiplier of ~1.2 to 1.5 usually maps sRPE to AU roughly.
        heuristic_load = (target_duration * target_rpe) * 1.4

        # 3. Blend them (60% Heuristic / 40% Model)
        # We weight the heuristic higher to ensure user inputs (Duration/RPE)
        # directly impact the visual output linearly.
        final_load = (heuristic_load * 0.6) + (raw_pred_load * 0.4)

        # --- B. Projected Risk Calculation ---

        # 1. Get AI Risk Probability
        if hasattr(risk_model, "predict_proba"):
            ai_risk_score = float(risk_model.predict_proba(features)[0][1])
        else:
            ai_risk_score = float(risk_model.predict(features)[0])

        # 2. Calculate Projected ACWR (Future State)
        # We estimate what the Rolling 7 & 28 will be *after* this session
        current_r7_sum = last_entry['rolling_7'] * 7
        current_r28_sum = last_entry['rolling_28'] * 28

        proj_rolling_7 = (current_r7_sum + final_load) / 8  # Approximate new avg
        proj_rolling_28 = (current_r28_sum + final_load) / 29

        proj_acwr = 0
        if proj_rolling_28 > 0:
            proj_acwr = proj_rolling_7 / proj_rolling_28

        # 3. Normalize Risk Factors (0.0 to 1.0)

        # Load Risk: 1000 AU is considered "Max/High"
        load_risk_factor = min(1.0, max(0.0, final_load / 1000.0))

        # ACWR Risk: > 1.5 is dangerous, > 2.0 is critical
        # We map 1.5 ACWR to roughly 0.8 risk score
        acwr_risk_factor = min(1.0, max(0.0, (proj_acwr - 0.8) / 1.2))

        # 4. Weighted Formula
        # 30% AI Model (Pattern recognition)
        # 30% Absolute Load (Tissue stress)
        # 40% Projected ACWR (Spike logic)
        w_ai = 0.30
        w_load = 0.30
        w_acwr = 0.40

        total_risk_score = (ai_risk_score * w_ai) + (load_risk_factor * w_load) + (acwr_risk_factor * w_acwr)
        risk_percentage = round(total_risk_score * 100, 1)

        # --- C. Labels & Visuals ---
        if risk_percentage < 35:
            risk_label = "Low"
            risk_color = "text-emerald-500"
        elif risk_percentage < 70:
            risk_label = "Moderate"
            risk_color = "text-amber-500"
        else:
            risk_label = "High"
            risk_color = "text-rose-500"

        calc_srpe = target_rpe * target_duration

        radar_data = [
            {"subject": "Load Impact", "A": 0, "B": min(100, (final_load / 1200) * 100)},
            {"subject": "ACWR Spike", "A": 0, "B": min(100, (proj_acwr / 2.0) * 100)},
            {"subject": "AI Pattern", "A": 0, "B": ai_risk_score * 100},
        ]

        return jsonify({
            "mech_load": round(final_load, 1),
            "srpe": round(calc_srpe, 1),
            "risk_percentage": risk_percentage,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "session_type": "High Volume" if target_duration > 80 else "Intensity Focus",
            "radar_data": radar_data,
            "projected_acwr": round(proj_acwr, 2)  # Helpful for debug
        }), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Model execution failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)