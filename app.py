import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Tytuł aplikacji w lewym górnym rogu
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: left;'>System Rekomendacji Filmów</h1>", unsafe_allow_html=True)

# Ścieżka do pliku CSV
csv_file_path = "Top_10000_Movies.csv"

try:
    # Wczytaj dane z pliku CSV
    movies_df = pd.read_csv(csv_file_path, low_memory=False, sep=',')

    # Wstępna obróbka danych
    movies_df['popularity'] = movies_df['popularity'].round(2)
    movies_df['vote_average'] = movies_df['vote_average'].round(2)
    movies_df['vote_count'] = movies_df['vote_count'].apply(pd.to_numeric, errors='coerce')  # Liczba głosów jako liczba
    movies_df['revenue'] = movies_df['revenue'].apply(pd.to_numeric, errors='coerce')  # Przychody jako liczba
    movies_df['runtime'] = movies_df['runtime'].apply(pd.to_numeric, errors='coerce')  # Czas trwania jako liczba
    movies_df.fillna(
        {'genre': 'Nieznane', 'release_date': 'Brak danych', 'original_language': 'Nieznany', 'overview': 'Brak opisu'},
        inplace=True)

    # Dodaj kolumnę z rokiem
    movies_df['release_year'] = movies_df['release_date'].apply(
        lambda x: x[:4] if isinstance(x, str) and len(x) >= 4 else 'Brak danych')

    # Definicje przedziałów dla list rozwijanych
    runtime_ranges = {
        "Wszystkie": (0, movies_df['runtime'].max()),
        "Krótkie (do 90 minut)": (0, 90),
        "Średnie (91-120 minut)": (91, 120),
        "Długie (121-150 minut)": (121, 150),
        "Bardzo długie (ponad 150 minut)": (151, movies_df['runtime'].max())
    }

    revenue_ranges = {
        "Wszystkie": (0, movies_df['revenue'].max()),
        "Małe (do 1 mln USD)": (0, 1_000_000),
        "Średnie (1-100 mln USD)": (1_000_001, 100_000_000),
        "Duże (101-500 mln USD)": (100_000_001, 500_000_000),
        "Ogromne (ponad 500 mln USD)": (500_000_001, movies_df['revenue'].max())
    }

    vote_average_ranges = {
        "Wszystkie": (0, movies_df['vote_count'].max()),
        "Niska (do 5)": (0, 4.99),
        "Średnia (5-7.5)": (5, 7.49),
        "Wysoka (powyżej 7.5)": (7.5, movies_df['vote_count'].max()),
        # "Bardzo wysoka (powyżej 9)": (9, movies_df['vote_count'].max())
    }

    filtered_df = movies_df.copy()

    # TF-IDF
    with st.container():
        search_query = st.text_input("Wpisz swoje zapytanie:", value="")
        
        # Copy the main DataFrame for filtering
        filtered_df = movies_df.copy()

    with st.container():
        # Rozkład filtrów w dwóch rzędach po trzy filtry
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        # Rząd 1
        with row1_col1:
            unique_years = sorted(filtered_df['release_year'].unique())
            selected_year = st.selectbox("Wybierz rok premiery:", ["Wszystkie"] + unique_years)
            if selected_year != "Wszystkie":
                filtered_df = filtered_df[filtered_df['release_year'] == selected_year].reset_index(drop=True)

        with row1_col2:
            unique_genres = sorted(
                set(genre for sublist in movies_df['genre'].dropna().apply(eval).tolist() for genre in sublist))
            selected_genre = st.selectbox("Wybierz gatunek:", ["Wszystkie"] + unique_genres)
            if selected_genre != "Wszystkie":
                filtered_df = filtered_df[filtered_df['genre'].apply(lambda x: selected_genre in eval(x))].reset_index(drop=True)

        with row1_col3:
            unique_languages = sorted(filtered_df['original_language'].unique())
            selected_language = st.selectbox("Wybierz język filmu:", ["Wszystkie"] + unique_languages)
            if selected_language != "Wszystkie":
                filtered_df = filtered_df[filtered_df['original_language'] == selected_language].reset_index(drop=True)

        # Rząd 2
        with row2_col1:
            selected_runtime_range = st.selectbox("Wybierz czas trwania filmu:", list(runtime_ranges.keys()))
            min_runtime, max_runtime = runtime_ranges[selected_runtime_range]
            filtered_df = filtered_df[(filtered_df['runtime'] >= min_runtime) & (filtered_df['runtime'] <= max_runtime)].reset_index(drop=True)

        with row2_col2:
            selected_revenue_range = st.selectbox("Wybierz zakres przychodu:", list(revenue_ranges.keys()))
            min_revenue, max_revenue = revenue_ranges[selected_revenue_range]
            filtered_df = filtered_df[(filtered_df['revenue'] >= min_revenue) & (filtered_df['revenue'] <= max_revenue)].reset_index(drop=True)

        with row2_col3:
            selected_vote_count_range = st.selectbox("Wybierz zakres średniej ocen:", list(vote_average_ranges.keys()))
            min_votes, max_votes = vote_average_ranges[selected_vote_count_range]
            filtered_df = filtered_df[
                (filtered_df['vote_average'] >= min_votes) & (filtered_df['vote_average'] <= max_votes)].reset_index(drop=True)

    # Handle TF-IDF and sorting based on search query
    if search_query:
        # Perform TF-IDF only on the filtered DataFrame
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(filtered_df['overview'])
        query_vector = vectorizer.transform([search_query]).toarray()
        similarity_scores = tfidf_matrix.toarray() @ query_vector.T
        similarity_scores = similarity_scores.flatten()

        # Add similarity column and sort by it
        filtered_df['similarity'] = similarity_scores
        filtered_df = filtered_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
    else:
        # If no query, add a default similarity score for consistent display
        filtered_df['similarity'] = 0

    # Wyświetl liczbę wyników
    st.write(f"Znaleziono {len(filtered_df)} filmów.")
    st.dataframe(
        filtered_df[['original_title', 'original_language', 'release_date', 'vote_average', 'popularity', 'similarity', 'runtime', 'revenue', 'genre', 'overview']].rename(
            columns={
                'original_title': 'Tytuł',
                'original_language': 'Język',
                'release_date': 'Data premiery',
                'vote_average': 'Średnia ocen',
                'popularity': 'Popularność',
                'similarity': 'Podobieństwo',
                'runtime': 'Czas trwania',
                'revenue': 'Przychód',
                'genre': 'Gatunek',
                'overview': 'Opis'
            }
        ),
        height=400,
        use_container_width=True
    )

except FileNotFoundError:
    st.error(f"Nie znaleziono pliku CSV pod ścieżką: {csv_file_path}.")
except Exception as e:
    st.error(f"Wystąpił błąd: {e}")