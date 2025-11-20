from flask import Flask, jsonify, request
from flask_cors import CORS
from dbconn import Database
import datetime

# ================= CONFIGURATION =================
app = Flask(__name__)
CORS(app)  # Enable CORS for Frontend communication

# Database Config (Update with your local credentials)
DB_CONFIG = {
    'dbname': 'smart-dashboard-db',
    'user': 'postgres',
    'password': 'carga',  # <--- CHANGE THIS
    'host': 'localhost',
    'port': '5500'
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
    query = "SELECT player_id, name, player_position, image_url FROM players ORDER BY name;"
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
# 3. PLAYER DEEP DIVE (Charts)
# -------------------------------------------------
@app.route('/api/player/<int:player_id>/trends', methods=['GET'])
def get_player_trends(player_id):
    """
    Get last 30 days of Load vs. Heart Rate for charts.
    Visualizes: Efficiency Decoupling.
    """
    query = """
        SELECT 
            session_id,
            activity_date,
            load_rpe as external_load,
            load_heart_rate as internal_cost,
            rpe_score,
            ac_ratio,
            efficiency_index
        FROM sessions
        WHERE player_id = %s
        ORDER BY activity_date ASC
        LIMIT 30;
    """
    data = db.execute(query, (player_id,), fetch_all=True)
    return jsonify(data if data else []), 200


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