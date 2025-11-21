from flask import Flask, jsonify, request
from flask_cors import CORS
from dbconn import Database
from datetime import datetime, timedelta

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


# -------------------------------------------------
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
# 2. DASHBOARD OVERVIEW (The "Main" Screen)
# -------------------------------------------------
@app.route('/api/dashboard/team-status', methods=['GET'])
def get_team_status():
    """
    Returns the latest readiness and load status for the whole team.
    Joins Players, Sessions (Load), and Wellness (Readiness).
    """
    # We get the data for the most recent date in the database
    query = """
        WITH LatestDate AS (
            SELECT MAX(activity_date) as max_date FROM sessions
        )
        SELECT 
            p.name,
            p.player_position,
            s.load_rpe,
            s.ac_ratio,
            s.efficiency_index,
            w.readiness_score,
            -- Simple Logic for Traffic Light Status
            CASE 
                WHEN s.ac_ratio > 1.5 OR w.readiness_score < 40 THEN 'RED'
                WHEN s.ac_ratio < 0.8 THEN 'YELLOW'
                ELSE 'GREEN'
            END as status_color
        FROM players p
        JOIN sessions s ON p.player_id = s.player_id
        LEFT JOIN daily_wellness w ON p.player_id = w.player_id AND s.activity_date = w.report_date
        JOIN LatestDate ld ON s.activity_date = ld.max_date
        ORDER BY status_color DESC, s.load_rpe DESC;
    """
    data = db.execute(query, fetch_all=True)
    return jsonify(data if data else []), 200

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
# 4. INPUT DATA (Wellness Check-In)
# -------------------------------------------------
@app.route('/api/wellness', methods=['POST'])
def submit_wellness():
    """
    Frontend sends: {player_id, sleep_quality, soreness, stress}
    We calculate readiness and insert.
    """
    data = request.json

    # Simple heuristic for Readiness Score (0-100)
    # (Sleep * 4) + (10 - Soreness * 2) + (10 - Stress * 2) ... simplified mapping
    # Assuming inputs are 1-10 for sleep, 1-5 for others
    readiness = (data['sleep_quality'] * 10 * 0.5) + ((6 - data['soreness']) * 10 * 0.25) + (
                (6 - data['stress']) * 10 * 0.25)

    query = """
        INSERT INTO daily_wellness 
        (player_id, report_date, sleep_quality, soreness_level, stress_level, readiness_score)
        VALUES (%s, CURRENT_DATE, %s, %s, %s, %s)
        RETURNING wellness_id;
    """
    params = (
        data['player_id'],
        data['sleep_quality'],
        data['soreness'],
        data['stress'],
        int(readiness)
    )

    result = db.execute(query, params, fetch_one=True)

    if result:
        return jsonify({"message": "Wellness logged", "id": result['wellness_id']}), 201
    return jsonify({"error": "Failed to log wellness"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)