import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.distance import Levenshtein
from textblob import TextBlob

# Tytuł aplikacji
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: left;'>System Rekomendacji Filmów</h1>", unsafe_allow_html=True)

# Ścieżka do pliku CSV
csv_file_path = "Top_10000_Movies.csv"

# Ustawienie stanu dla widoczności sekcji generowania opisu
if 'show_description' not in st.session_state:
    st.session_state['show_description'] = False

try:
    # Wczytaj dane z pliku CSV
    movies_df = pd.read_csv(csv_file_path, low_memory=False, sep=',')

    # Wstępna obróbka danych
    movies_df['popularity'] = movies_df['popularity'].round(2)
    movies_df['vote_average'] = movies_df['vote_average'].round(2)
    movies_df['vote_count'] = movies_df['vote_count'].apply(pd.to_numeric, errors='coerce')
    movies_df['revenue'] = movies_df['revenue'].apply(pd.to_numeric, errors='coerce')
    movies_df['runtime'] = movies_df['runtime'].apply(pd.to_numeric, errors='coerce')
    movies_df.fillna(
        {'genre': 'Nieznane', 'release_date': 'Brak danych', 'original_language': 'Nieznany', 'overview': 'Brak opisu'},
        inplace=True)

    # Dodaj kolumnę z rokiem
    movies_df['release_year'] = movies_df['release_date'].apply(
        lambda x: x[:4] if isinstance(x, str) and len(x) >= 4 else 'Brak danych')

    # Analiza tonu
    def analyze_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    movies_df['sentiment'] = movies_df['overview'].apply(analyze_sentiment)
    movies_df['tone'] = movies_df['sentiment'].apply(
        lambda x: 'Pozytywny' if x > 0 else 'Negatywny' if x < 0 else 'Neutralny'
    )

    filtered_df = movies_df.copy()

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
        "Wszystkie": (0, movies_df['vote_average'].max()),
        "Niska (do 5)": (0, 4.99),
        "Średnia (5-7.5)": (5, 7.49),
        "Wysoka (powyżej 7.5)": (7.5, movies_df['vote_average'].max())
    }

    # Generowanie słownika poprawnych słów z opisów
    valid_words = set(word.lower() for desc in movies_df['overview'] for word in str(desc).split())

    # Funkcja autokorekty zapytania
    def correct_query(query, valid_words, threshold=3):
        distances = [(word, Levenshtein.distance(query.lower(), word.lower())) for word in valid_words]
        distances.sort(key=lambda x: x[1])
        best_match, best_distance = distances[0]
        return best_match if best_distance <= threshold else query

    # Wprowadzenie zapytania przez użytkownika
    with st.container():
        search_query = st.text_input("Wpisz swoje zapytanie:", value="")
        corrected_query = correct_query(search_query, valid_words) if search_query else ""

        if corrected_query != search_query:
            st.info(f"Twoje zapytanie zostało poprawione na: {corrected_query}")

    similarity_method = st.radio("Wybierz miarę podobieństwa:", ("Miara cosinusa", "LSI"))

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
        selected_vote_average_range = st.selectbox("Wybierz zakres średniej ocen:", list(vote_average_ranges.keys()))
        min_vote_avg, max_vote_avg = vote_average_ranges[selected_vote_average_range]
        filtered_df = filtered_df[
            (filtered_df['vote_average'] >= min_vote_avg) & (filtered_df['vote_average'] <= max_vote_avg)].reset_index(drop=True)

    # Filtr tonu
    tone_filter = st.selectbox("Wybierz ton opisu filmu:", ("Wszystkie", "Pozytywny", "Neutralny", "Negatywny"))
    if tone_filter != "Wszystkie":
        filtered_df = filtered_df[filtered_df['tone'] == tone_filter]

    filtered_df['similarity'] = 0

    # Obsługa metod podobieństwa
    if search_query:
        # Autokorekta zawsze działa na zapytaniu
        corrected_query = correct_query(search_query, valid_words)

        if similarity_method == "Miara cosinusa":
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(filtered_df['overview'])
            query_vector = vectorizer.transform([corrected_query]).toarray()  # Użycie poprawionego zapytania
            tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2')
            query_vector_normalized = normalize(query_vector, norm='l2')
            cosine_similarity_scores = tfidf_matrix_normalized @ query_vector_normalized.T
            cosine_similarity_scores = cosine_similarity_scores.flatten()
            filtered_df['similarity'] = cosine_similarity_scores

        elif similarity_method == "LSI":
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(filtered_df['overview'])
            query_vector = vectorizer.transform([corrected_query]).toarray()  # Użycie poprawionego zapytania
            svd = TruncatedSVD(n_components=100, random_state=42)
            lsi_matrix = svd.fit_transform(tfidf_matrix)
            query_lsi_vector = svd.transform(query_vector)
            lsi_similarity_scores = cosine_similarity(lsi_matrix, query_lsi_vector)
            lsi_similarity_scores = lsi_similarity_scores.flatten()
            filtered_df['similarity'] = lsi_similarity_scores

        filtered_df = filtered_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    # Wyświetl liczbę wyników
    st.write(f"Znaleziono {len(filtered_df)} filmów.")
    st.dataframe(
        filtered_df[['original_title', 'original_language', 'release_date', 'vote_average', 'popularity', 'similarity', 'runtime', 'revenue', 'genre', 'overview', 'tone']].rename(
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
                'overview': 'Opis',
                'tone': 'Ton'
            }
        ),
        height=400,
        use_container_width=True
    )

    # Przycisk do przełączania widoczności sekcji generowania opisu
    if st.button("Przejdź do generowania opisu"):
        st.session_state['show_description'] = not st.session_state['show_description']

    # Sekcja generowania opisu, widoczna tylko jeśli aktywowana
    if st.session_state['show_description']:
        st.markdown("---")
        st.subheader("Generowanie opisu na podstawie reguł")

        selected_movie = st.selectbox("Wybierz film do generowania opisu:", filtered_df['original_title'])

        if selected_movie:
            selected_row = filtered_df[filtered_df['original_title'] == selected_movie].iloc[0]

            def generate_description(movie_row):
                description = f"Tytuł: {movie_row['original_title']}\n"
                description += f"Rok wydania: {movie_row['release_year']}\n"
                if movie_row['tone'] == "Pozytywny":
                    description += "Opis filmu wskazuje na pozytywny wydźwięk.\n"
                elif movie_row['tone'] == "Negatywny":
                    description += "Opis filmu wskazuje na mroczny lub negatywny ton.\n"
                else:
                    description += "Opis filmu jest neutralny w tonie.\n"

                if 'runtime' in movie_row and not pd.isna(movie_row['runtime']):
                    if movie_row['runtime'] < 90:
                        description += "To krótki film, idealny na szybki seans.\n"
                    elif movie_row['runtime'] <= 120:
                        description += "To średniej długości film.\n"
                    else:
                        description += "To długi film, oferujący rozbudowaną historię.\n"

                if 'revenue' in movie_row and not pd.isna(movie_row['revenue']):
                    if movie_row['revenue'] < 1_000_000:
                        description += "Film wygenerował małe przychody.\n"
                    elif movie_row['revenue'] <= 100_000_000:
                        description += "Film osiągnął średnie wyniki finansowe.\n"
                    else:
                        description += "Film odniósł ogromny sukces finansowy.\n"

                return description

            st.text(generate_description(selected_row))

except FileNotFoundError:
    st.error(f"Nie znaleziono pliku CSV pod ścieżką: {csv_file_path}.")
except Exception as e:
    st.error(f"Wystąpił błąd: {e}")
