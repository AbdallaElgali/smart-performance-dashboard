import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_config = {

}


class Database:
    def __init__(self, db_config):
        """
        Initialize with database configuration dictionary.
        Example: {'dbname': 'basketball_db', 'user': 'postgres', ...}
        """
        self.config = db_config

    def get_connection(self):
        """Creates a new database connection."""
        try:
            conn = psycopg2.connect(**self.config)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def execute(self, query, params=None, fetch_all=False, fetch_one=False):
        """
        Executes a query and returns results or row count.

        Args:
            query (str): SQL query to execute.
            params (tuple/dict, optional): Parameters to bind to the query.
            fetch_all (bool): If True, returns all rows (SELECT).
            fetch_one (bool): If True, returns a single row (SELECT).

        Returns:
            - List of dicts (if fetch_all=True)
            - Single dict (if fetch_one=True)
            - None (if no rows found)
            - Row count (if INSERT/UPDATE/DELETE)
        """
        conn = None
        cursor = None
        result = None

        try:
            conn = self.get_connection()
            # RealDictCursor ensures results come back as {'column': value}
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(query, params)

            if fetch_all:
                result = cursor.fetchall()
            elif fetch_one:
                result = cursor.fetchone()
            else:
                # For INSERT/UPDATE, commit the transaction
                conn.commit()
                result = cursor.rowcount

            return result

        except psycopg2.Error as e:
            logger.error(f"Database Error: {e}")
            if conn:
                conn.rollback()
            return None  # Or raise e if you want the app to crash/handle it

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()