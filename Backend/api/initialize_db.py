import pandas as pd
import numpy as np
from datetime import datetime
import io
import logging
from dbconn import Database  # Assuming dbconn.py is in the same directory

logger = logging.getLogger(__name__)


class DataIngestor:
    def __init__(self, db_config):
        self.db = Database(db_config)

    def process_csv_and_ingest(self, csv_data):
        """
        Processes a single CSV string, handles player insertion, and inserts session data.

        Args:
            csv_data (str): The raw CSV content string.
        """
        try:
            # 1. READ DATA
            df = pd.read_csv(csv_data)

            # Clean columns and types
            df.columns = df.columns.str.strip()
            # FIX: Ensure all dates are valid objects
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])

            # Filter out entirely zero/rest rows for sessions to avoid cluttering the DB
            df_active = df[(df['LoadRPE'] > 0) | (df['LoadFC'] > 0)].copy()

            if df_active.empty:
                logger.info("No active sessions detected in this file.")
                return 0

            print(df_active['Player'].unique())
            # 2. PLAYER MANAGEMENT (UPSERT)
            player_map = self._get_or_create_players(df_active['Player'].unique())
            logger.info('Player created: ', player_map)
            df_active['player_id'] = df_active['Player'].map(player_map)

            # 3. SPLIT SESSIONS (RPEm and RPEt) AND INGEST
            session_count = 0

            # Iterate through each row of active data
            for _, row in df_active.iterrows():
                player_id = row['player_id']

                # CRITICAL GUARD: Only insert if we successfully found a player ID
                if pd.isna(player_id):
                    logger.warning(f"Skipping row for Player {row['Player']}: ID not found or created.")
                    continue

                # --- Morning Session (RPEm) ---
                if row['RPEm'] > 0:
                    self._insert_session(row, player_id, 'AM', 'RPEm')  # Renamed from RPEm to AM/PM for session_type
                    session_count += 1

                # --- Evening/Total Session (RPEt) ---
                # Logic: If RPEt > 0 AND (RPEt != RPEm OR RPEm == 0), log PM session.
                if row['RPEt'] > 0 and (row['RPEt'] != row['RPEm']):
                    self._insert_session(row, player_id, 'PM', 'RPEt')  # Renamed from RPEt to AM/PM for session_type
                    session_count += 1

            logger.info(f"Successfully ingested {session_count} new session records.")
            return session_count

        except Exception as e:
            logger.error(f"Failed during data processing: {e}", exc_info=True)
            return 0

    def _get_or_create_players(self, player_names):
        """
        Ensures all players exist in the 'players' table and returns a name-to-ID map.
        Uses the PostgreSQL standard 'ON CONFLICT DO UPDATE' pattern for robust upserting.
        """
        player_map = {}
        for i, name in enumerate(player_names):
            # PostgreSQL UPSERT approach: Try to insert, if conflict (name exists), do nothing, and RETURNING the ID anyway.
            # This handles both creation and retrieval in one atomic, safe operation.
            upsert_query = """
                INSERT INTO players (name, player_id) VALUES (%s, %s) ON CONFLICT DO NOTHING
            """
            p_id = i + 1
            # Execute the upsert query and fetch the resulting player_id
            row_count = self.db.execute(upsert_query, (name,p_id))

            if row_count:
                player_map[name] = p_id
            else:
                # Fallback if the upsert didn't return an ID (should only happen if transaction failed entirely)
                logger.warning(f"Failed critical upsert operation for player: {name}. Skipping.")

        return player_map

    def _insert_session(self, row, player_id, session_type, rpe_col_name):
        """
        Inserts a single session row into the 'sessions' table.
        Adjusted to map to the corrected schema from the prior step.
        """
        rpe_score = float(row[rpe_col_name])

        # Calculate dummy duration
        estimated_duration = float(row['LoadRPE']) / rpe_score if rpe_score > 0 and float(row['LoadRPE']) > 0 else 0



        query = """
            INSERT INTO sessions (
                player_id, activity_date, 
                duration_min, total_player_load, load_rpe, load_heart_rate, 
                rpe_score, ac_ratio
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """
        params = (
            int(player_id),  # Ensure player_id is INTEGER
            row['Date'],
            estimated_duration,
            row['LoadRPE'],
            row['LoadRPE'],
            row['LoadFC'],
            rpe_score,
            row["AC_Ratio"]  # Use the safe calculated version
        )

        self.db.execute(query, params)


# =======================================================================
# EXECUTION BLOCK FOR TESTING AND DATA INGESTION
# =======================================================================

# Database Config (Use your actual configuration when running)
DB_CONFIG = {
    'dbname': 'smart-dashboard-db',
    'user': 'postgres',
    'password': 'carga',  # <--- CHANGE THIS
    'host': 'localhost',
    'port': '5500'
}
def ini():


    # Sample data provided by the user
    CSV_DATA = "./data/full_athlete_metrics.csv"

    logger.info("Starting data ingestion process...")

    # Instantiate the Ingestor class
    ingestor = DataIngestor(DB_CONFIG)

    # Run the ingestion process with the sample data
    inserted_count = ingestor.process_csv_and_ingest(CSV_DATA)

    if inserted_count > 0:
        logger.info(f"Ingestion successful. Total sessions inserted: {inserted_count}")
    else:
        logger.warning("Ingestion finished with 0 new active sessions inserted.")

ini()