-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS alexis;

-- Connect to the database
\c alexis;

-- Create the chat_history table if it doesn't exist
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    chat_id INTEGER,
    message TEXT,
    type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the chat_id_seq sequence if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_sequences WHERE schemaname = 'public' AND sequencename = 'chat_id_seq') THEN
        CREATE SEQUENCE chat_id_seq START WITH 1 INCREMENT BY 1 NO MAXVALUE NO CYCLE;
    END IF;
END
$$;