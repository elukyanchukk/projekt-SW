from functions.text_functions import analyze_sentiment
import sqlite3
import pandas as pd


# ścieżka do bazy danych SQLite
DB_PATH = "movies.db"


# pobieranie danych z bazy
def fetch_movies(query="SELECT * FROM movies", params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    df['sentiment'] = df['overview'].apply(analyze_sentiment)
    df['tone'] = df['sentiment'].apply(
        lambda x: 'Pozytywny' if x > 0 else 'Negatywny' if x < 0 else 'Neutralny'
    )

    return df


# inicjalizacja bazy danych
def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_title TEXT,
            popularity REAL,
            vote_average REAL,
            vote_count INTEGER,
            revenue REAL,
            runtime REAL,
            genre TEXT,
            release_date TEXT,
            original_language TEXT,
            overview TEXT,
            release_year TEXT,
            sentiment REAL,
            tone TEXT,
            vote_category INTEGER
        )
    """)

    conn.commit()
    conn.close()