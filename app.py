import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

from functions import analyze_sentiment, correct_query, generate_description, preprocess_df, create_knn_model, classify_vote_category, create_word_cloud, create_bar_chart

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

    movies_df = preprocess_df(movies_df)

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

    # Wprowadzenie zapytania przez użytkownika
    with st.container():
        search_query = st.text_input("Wpisz swoje zapytanie:", value="")
        corrected_query = correct_query(search_query, valid_words) if search_query else ""

        if corrected_query != search_query:
            st.info(f"Twoje zapytanie zostało poprawione na: {corrected_query}")

    similarity_method = st.radio("Wybierz miarę podobieństwa:", ("Miara cosinusa", "Miara Jaccarda", "LSI"))

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

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(filtered_df['overview'])
        query_vector = vectorizer.transform([corrected_query]).toarray()  # Użycie poprawionego zapytania

        if similarity_method == "Miara cosinusa":
            tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2')
            query_vector_normalized = normalize(query_vector, norm='l2')
            cosine_similarity_scores = tfidf_matrix_normalized @ query_vector_normalized.T
            cosine_similarity_scores = cosine_similarity_scores.flatten()
            filtered_df['similarity'] = cosine_similarity_scores

        elif similarity_method == "Miara Jaccarda":
            count_vectorizer = CountVectorizer(binary=True, stop_words='english')
            binary_matrix = count_vectorizer.fit_transform(filtered_df['overview']).toarray()
            query_binary_vector = count_vectorizer.transform([corrected_query]).toarray()
            jaccard_scores = [
                jaccard_score(binary_matrix[i], query_binary_vector[0], average='binary')
                for i in range(binary_matrix.shape[0])
            ]

            filtered_df['similarity'] = jaccard_scores

        elif similarity_method == "LSI":
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

            st.text(generate_description(selected_row))

    # Wprowadzenie informacji o nowym filmie i przewidywanie, czy ocena będzie wysoka, średnia lub niska
    with st.container(): 
        st.subheader("Przewidywanie oceny filmu")
        st.write("Wprowadź informacje o filmie, żeby się dowiedzieć, czy on będzie dobry, średni czy zły")

        title = st.text_input("Tytuł filmu", placeholder="Wprowadź tytuł filmu")
        overview = st.text_area("Opis filmu", placeholder="Wprowadź szczegółowy opis filmu", height=68)

        classification_df = movies_df[['original_title', 'overview', 'vote_category']]

        model = create_knn_model(classification_df)

        if st.button("Przewiduj kategorię oceny"):
            predicted_category, nearest_neighbors_df = classify_vote_category(model, title, overview, classification_df)
            st.write(f"Na podstawie 10 filmów najbardziej podobnych do tego, film najprawdopodobniej będzie **{predicted_category}**.")

            st.dataframe(nearest_neighbors_df[['original_title', 'overview', 'vote_category']].rename(
                columns ={'original_title': 'Tytuł',
                          'overview': 'Opis', 
                          'vote_category': 'Kategoria ocen'
                          }
            ),
                height=400,
                use_container_width=True
            )

            plot_1, plot_2 = st.columns(2)

            with plot_1:
                st.write("Chmura słów na podstawie opisów filmów")
                create_word_cloud(nearest_neighbors_df)
                
            with plot_2:
                st.write("10 najczęściej pojawiających się słów i ich liczba")
                create_bar_chart(nearest_neighbors_df)
                


except FileNotFoundError:
    st.error(f"Nie znaleziono pliku CSV pod ścieżką: {csv_file_path}.")
except Exception as e:
    st.error(f"Wystąpił błąd: {e}")
