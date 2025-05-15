import psycopg
import logging
from app.config import DB_URI
import os

def get_db_uri():
    """Get database URI based on environment."""
    if os.environ.get("TESTING", "False").lower() == "true":
        # Use test database when in testing mode
        from app.config import DB_USER, DB_PASSWORD, DB_HOST
        test_db = os.environ.get("POSTGRES_DB", "alexis_test")
        return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{test_db}"
    else:
        # Use default DB_URI from config
        return DB_URI


def initialize_database():
    """Initialize the database, creating tables and sequences if they don't exist."""
    try:
        # Get appropriate DB URI based on environment
        db_uri = get_db_uri()
        
        with psycopg.connect(db_uri) as conn:
            with conn.cursor() as cur:
                # Check if chat_history table exists
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'chat_history'
                    );
                """
                )
                table_exists = cur.fetchone()[0]

                # Create table if it doesn't exist
                if not table_exists:
                    logging.info("Creating chat_history table...")
                    cur.execute(
                        """
                        CREATE TABLE chat_history (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT,
                            chat_id INTEGER,
                            message TEXT,
                            type TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    )

                # Check if sequence exists
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM pg_sequences
                        WHERE schemaname = 'public'
                        AND sequencename = 'chat_id_seq'
                    );
                """
                )
                sequence_exists = cur.fetchone()[0]

                # Create sequence if it doesn't exist
                if not sequence_exists:
                    logging.info("Creating chat_id_seq sequence...")
                    # Get the maximum chat_id to start the sequence from
                    cur.execute(
                        "SELECT COALESCE(MAX(chat_id), 0) + 1 FROM chat_history"
                    )
                    next_chat_id = cur.fetchone()[0]
                    # Creating sequence based on the last number
                    cur.execute(
                        f"""
                        CREATE SEQUENCE chat_id_seq
                        START WITH {next_chat_id}
                        INCREMENT BY 1
                        NO MAXVALUE
                        NO CYCLE
                    """
                    )

                conn.commit()
                logging.info("Database initialization completed successfully")

    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise


def get_next_chat_id(connection):
    """Get the next chat ID from the sequence."""
    cursor = connection.cursor()
    cursor.execute("SELECT nextval('chat_id_seq')")
    next_chat_id = cursor.fetchone()[0]
    cursor.close()
    return next_chat_id


def load_session_mapping():
    """Load the mapping of session IDs to chat IDs from the database."""
    try:
        # Get appropriate DB URI based on environment
        db_uri = get_db_uri()
        
        with psycopg.connect(db_uri) as conn:
            with conn.cursor() as cur:
                # Get the most recent session for each chat_id
                cur.execute(
                    """
                    SELECT DISTINCT ON (chat_id) session_id, chat_id, created_at
                    FROM chat_history
                    ORDER BY chat_id, created_at DESC
                """
                )
                rows = cur.fetchall()
                session_mapping = {}
                for session_id, chat_id, created_at in rows:
                    timestamp = created_at.timestamp()
                    session_mapping[session_id] = {
                        "chat_id": chat_id,
                        "timestamp": timestamp,
                        "connector_selector": None,
                    }
                return session_mapping
    except Exception as e:
        logging.error(f"Error loading session mapping: {str(e)}")
        return {}


def get_db_connection():
    """Get a database connection."""
    # Get appropriate DB URI based on environment
    db_uri = get_db_uri()
    return psycopg.connect(db_uri)
