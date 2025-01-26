import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

import functions.db_functions as db
import functions.text_functions as txt
import functions.similarity_functions as sim
import functions.classification_functions as clf
import functions.plot_functions as plt


# konfiguracja strony
st.set_page_config(layout="wide")

# ścieżka do bazy danych SQLite

db.initialize_database()

# nazwa aplikacji

st.markdown("""
    <div style='background-color: #f5c518; border-radius: 5px; padding: 10px 20px; display: inline-block; text-align: center; margin: 0 auto;margin-bottom: 20px;'>
        <h1 style='color: #000; font-family: Arial, sans-serif; font-weight: bold; margin: 0; text-align: center;'>
            System Rekomendacji Filmów
        </h1>
    </div>
""", unsafe_allow_html=True)

try:
    movies_df = db.fetch_movies()
    filtered_df = movies_df.copy()

    # słownik do autokorekty
    valid_words = set(word.lower() for desc in movies_df['overview'] for word in str(desc).split())

    st.markdown("""
        <style>
        /* Stylizacja dla nagłówka "Wyszukiwarka" */
        h2 {
            text-align: left;
            font-size: 2em;
            margin-bottom: 0.5em; /* Odstęp pod nagłówkiem */
        }

        /* Stylizacja dla żółtej kreski */
        h2::after {
            content: "";
            display: block;
            width: 100%;
            height: 2px;
            background-color: #f5c518; /* Kolor żółty */
            margin-top: 5px; /* Odstęp od nagłówka */
            margin-bottom: 10px; /* Większy odstęp od treści poniżej */
        }
        </style>
        <h2>Wyszukiwarka</h2>
    """, unsafe_allow_html=True)

    search_query = st.text_input("Wpisz swoje zapytanie:", value="")
    corrected_query = txt.correct_query(search_query, valid_words) if search_query else ""

    if corrected_query != search_query:
        st.info(f"Twoje zapytanie zostało poprawione na: {corrected_query}")

    similarity_method = st.radio("Wybierz miarę podobieństwa:", ("Miara cosinusa", "LSI", "Miara Jaccarda"))

    # dodanie kolumny 'similarity'
    filtered_df['similarity'] = 0  

    # metoda podobieństw
    if search_query:

        corrected_query = txt.correct_query(search_query, valid_words)

        vectorizer = TfidfVectorizer(stop_words='english', preprocessor=txt.preprocess_text)
        tfidf_matrix = vectorizer.fit_transform(filtered_df['overview'])
        query_vector = vectorizer.transform([corrected_query]).toarray()  

        if similarity_method == "Miara cosinusa":
            cosine_similarity_scores = sim.cosine(tfidf_matrix, query_vector)
            filtered_df['similarity'] = cosine_similarity_scores

        elif similarity_method == "Miara Jaccarda":
            jaccard_scores = sim.jaccard(filtered_df, corrected_query)
            filtered_df['similarity'] = jaccard_scores

        elif similarity_method == "LSI":
            lsi_similarity_scores = sim.lsi(tfidf_matrix, query_vector)
            filtered_df['similarity'] = lsi_similarity_scores

        filtered_df = filtered_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    with st.container():
        st.markdown("""
            <h2 style='
                width: 100%; 
                color: #FFFFFF; /* Kolor tekstu */
                padding-bottom: 10px; /* Odstęp pomiędzy tekstem a linią */
                margin-bottom: 20px; /* Odstęp pod całą sekcją */
                text-align: left;'>
                Filtry
            </h2>
        """, unsafe_allow_html=True)
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)


        with row1_col1:
            unique_years = sorted(filtered_df['release_year'].unique())
            selected_year = st.selectbox("Wybierz rok premiery:", ["Wszystkie"] + unique_years)
            if selected_year != "Wszystkie":
                filtered_df = filtered_df[filtered_df['release_year'] == selected_year].reset_index(drop=True)

        with row1_col2:
            unique_genres = sorted(set(genre for sublist in movies_df['genre'].dropna().apply(eval).tolist() for genre in sublist))
            selected_genre = st.selectbox("Wybierz gatunek:", ["Wszystkie"] + unique_genres)
            if selected_genre != "Wszystkie":
                filtered_df = filtered_df[filtered_df['genre'].apply(lambda x: selected_genre in eval(x))].reset_index(drop=True)

        with row1_col3:
            unique_languages = sorted(filtered_df['original_language'].unique())
            selected_language = st.selectbox("Wybierz język filmu:", ["Wszystkie"] + unique_languages)
            if selected_language != "Wszystkie":
                filtered_df = filtered_df[filtered_df['original_language'] == selected_language].reset_index(drop=True)

        with row2_col1:
            runtime_ranges = {
                "Wszystkie": (0, movies_df['runtime'].max()),
                "Krótkie (do 90 minut)": (0, 90),
                "Średnie (91-120 minut)": (91, 120),
                "Długie (121-150 minut)": (121, 150),
                "Bardzo długie (ponad 150 minut)": (151, movies_df['runtime'].max())
            }
            selected_runtime_range = st.selectbox("Wybierz czas trwania filmu:", list(runtime_ranges.keys()))
            min_runtime, max_runtime = runtime_ranges[selected_runtime_range]
            filtered_df = filtered_df[(filtered_df['runtime'] >= min_runtime) & (filtered_df['runtime'] <= max_runtime)].reset_index(drop=True)

        with row2_col2:
            revenue_ranges = {
                "Wszystkie": (0, movies_df['revenue'].max()),
                "Małe (do 1 mln USD)": (0, 1_000_000),
                "Przeciętne (1-100 mln USD)": (1_000_001, 100_000_000),
                "Duże (101-500 mln USD)": (100_000_001, 500_000_000),
                "Ogromne (ponad 500 mln USD)": (500_000_001, movies_df['revenue'].max())
            }
            selected_revenue_range = st.selectbox("Wybierz zakres przychodu:", list(revenue_ranges.keys()))
            min_revenue, max_revenue = revenue_ranges[selected_revenue_range]
            filtered_df = filtered_df[(filtered_df['revenue'] >= min_revenue) &
                                      (filtered_df['revenue'] <= max_revenue)].reset_index(drop=True)

        with row2_col3:
            vote_average_ranges = {
                "Wszystkie": (0, movies_df['vote_average'].max()),
                "Niska (do 5)": (0, 4.99),
                "Przeciętna (5-7.5)": (5, 7.49),
                "Wysoka (powyżej 7.5)": (7.5, movies_df['vote_average'].max())
            }
            selected_vote_average_range = st.selectbox("Wybierz zakres średniej ocen:", list(vote_average_ranges.keys()))
            min_vote_avg, max_vote_avg = vote_average_ranges[selected_vote_average_range]
            filtered_df = filtered_df[(filtered_df['vote_average'] >= min_vote_avg) & (filtered_df['vote_average'] <= max_vote_avg)].reset_index(drop=True)


        tone_filter = st.selectbox("Wybierz wydźwięk opisu filmu:", ["Wszystkie", "Pozytywny", "Neutralny", "Negatywny"])
        if tone_filter != "Wszystkie":
            filtered_df = filtered_df[filtered_df['tone'] == tone_filter].reset_index(drop=True)

    # tabela wyników
    st.markdown("""
        <h2 style='
            width: 100%; 
            color: #FFFFFF; /* Kolor tekstu */
            padding-bottom: 10px; /* Odstęp pomiędzy tekstem a linią */
            margin-bottom: 20px; /* Odstęp pod całą sekcją */
            text-align: left;'>
            Wynik filtrowania
        </h2>
    """, unsafe_allow_html=True)
    st.write(f"Znaleziono {len(filtered_df)} filmów.")
    st.dataframe(
        filtered_df[['similarity','id', 'original_title', 'original_language', 'release_date','release_year',
                     'vote_average','vote_count', 'popularity', 'runtime', 'revenue', 'genre', 'overview',
                      'tagline', 'tone']].rename(
            columns={
                'similarity': 'Podobieństwo',
                'id': 'ID',
                'original_title': 'Tytuł',
                'original_language': 'Język',
                'release_date': 'Data premiery',
                'release_year': 'Rok premiery',
                'vote_average': 'Średnia ocen',
                'vote_count': 'Liczba głosów',
                'popularity': 'Popularność',
                'runtime': 'Czas trwania [min]',
                'revenue': 'Przychód [mln]',
                'genre': 'Gatunek',
                'overview': 'Opis',
                'tagline': 'Tagline',
                'tone': 'Wydźwięk'
            }
        ),
        height=400,
        use_container_width=True
    )
    st.markdown("""
        <h2 style='
            width: 100%; 
            color: #FFFFFF; /* Kolor tekstu */
            padding-bottom: 10px; /* Odstęp pomiędzy tekstem a linią */
            margin-bottom: 20px; /* Odstęp pod całą sekcją */
            text-align: left;'>
            Krótki opis filmu
        </h2>
    """, unsafe_allow_html=True)
    selected_movie = st.selectbox("Wybierz film do generowania opisu:", filtered_df['original_title'])
    if selected_movie:
        selected_row = filtered_df[filtered_df['original_title'] == selected_movie].iloc[0]
        st.text(txt.generate_description(selected_row))

    # Sekcja przewidywania kategorii ocen filmu
    st.markdown("""
        <h2 style='
            width: 100%; 
            color: #FFFFFF; /* Kolor tekstu */
            padding-bottom: 10px; /* Odstęp pomiędzy tekstem a linią */
            margin-bottom: 20px; /* Odstęp pod całą sekcją */
            text-align: left;'>
            Przewidywanie oceny filmu
        </h2>
    """, unsafe_allow_html=True)

    # Wprowadzenie tytułu i opisu filmu
    title = st.text_input("Tytuł filmu", placeholder="Wprowadź tytuł filmu")
    overview = st.text_area("Opis filmu", placeholder="Wprowadź szczegółowy opis filmu", height=68)

    # Przygotowanie danych i modeli
    classification_df = movies_df[['original_title', 'overview', 'vote_category']]
    knn_model = clf.create_knn_model(classification_df)
    svm_model = clf.create_svm_model(classification_df)

    # Wybór metody klasyfikacji
    classification_method = st.radio(
        "Wybierz metodę klasyfikacji:",
        ("KNN", "SVM")
    )

    # Przycisk przewidywania kategorii ocen
    if st.button("Przewiduj kategorię oceny", key="predict_button"):
        if title and overview:
            # Wywołanie odpowiedniej metody klasyfikacji
            if classification_method == "KNN":
                predicted_category, nearest_neighbors_df = clf.classify_with_knn(knn_model, title, overview,
                                                                             classification_df)

                # Wyświetlanie wyników dla KNN
                st.write(
                    f"Na podstawie 10 filmów najbardziej podobnych do tego, film najprawdopodobniej będzie **{predicted_category}**."
                )
                st.dataframe(
                    nearest_neighbors_df[['original_title', 'overview', 'vote_category']].rename(
                        columns={
                            'original_title': 'Tytuł',
                            'overview': 'Opis',
                            'vote_category': 'Kategoria ocen'
                        }
                    ),
                    height=400,
                    use_container_width=True
                )

                # Wizualizacje: chmura słów i wykres słupkowy
                plot_1, plot_2 = st.columns(2)

                with plot_1:
                    st.write("Chmura słów na podstawie opisów filmów")
                    plt.create_word_cloud(nearest_neighbors_df)

                with plot_2:
                    st.write("10 najczęściej pojawiających się słów i ich liczba")
                    plt.create_bar_chart(nearest_neighbors_df)

            elif classification_method == "SVM":
                predicted_category = clf.classify_with_svm(svm_model, title, overview)

                # Wyświetlanie wyników dla SVM
                st.write(f"Film najprawdopodobniej będzie w kategorii **{predicted_category}**.")
        else:
            st.error("Proszę wypełnić wszystkie pola!")

except Exception as e:
    st.error(f"Wystąpił błąd podczas przetwarzania danych: {e}")
