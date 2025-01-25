from textblob import TextBlob
from rapidfuzz.distance import Levenshtein
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

def preprocess_df(movies_df):
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

    movies_df['sentiment'] = movies_df['overview'].apply(analyze_sentiment)
    movies_df['tone'] = movies_df['sentiment'].apply(
        lambda x: 'Pozytywny' if x > 0 else 'Negatywny' if x < 0 else 'Neutralny'
    )

    movies_df['vote_category'] = pd.qcut(
        movies_df['vote_average'],
        q=5,
        labels=[1, 2, 3, 4, 5]
    )

    movies_df = movies_df.drop_duplicates(subset=['original_title', 'release_year', 'overview'], keep='first')
    movies_df.reset_index(drop=True, inplace=True)

    return movies_df

# analiza sentymentu
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


# odległość levenshteina - autokorekta
def correct_query(query, valid_words, threshold=3):
    distances = [(word, Levenshtein.distance(query.lower(), word.lower())) for word in valid_words]
    distances.sort(key=lambda x: x[1])
    best_match, best_distance = distances[0]
    return best_match if best_distance <= threshold else query


# generowanie krótkiego opisu
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

def create_knn_model(classification_df):
    classification_df = classification_df.copy()
    classification_df['text'] = classification_df['original_title'] + " " + classification_df['overview']

    X = classification_df['text']
    y = classification_df['vote_category']

    model = make_pipeline(
        TfidfVectorizer(),
        KNeighborsClassifier(n_neighbors=10)
    )

    model.fit(X, y)

    return model



# klasyfikacja dokumentów - knn
def classify_vote_category(model, user_title, user_overview, classification_df):
    category_mapping = {
        1: "bardzo zły",
        2: "zły",
        3: "średni",
        4: "dobry",
        5: "bardzo dobry"
    }

    user_text = user_title + " " + user_overview

    try:
        predicted_numeric_category = model.predict([user_text])[0]
        predicted_numeric_category = int(predicted_numeric_category)

        predicted_category = category_mapping[predicted_numeric_category]

        distances, indices = model.named_steps['kneighborsclassifier'].kneighbors(
            model.named_steps['tfidfvectorizer'].transform([user_text]), n_neighbors=10
        )

        nearest_neighbors_df = classification_df.iloc[indices[0]].reset_index(drop=True)
        nearest_neighbors_df['vote_category'] = nearest_neighbors_df['vote_category'].astype(int)

        nearest_neighbors_df['vote_category'] = nearest_neighbors_df['vote_category'].map(category_mapping).fillna("Nieznana kategoria")

        return predicted_category, nearest_neighbors_df
    except Exception as e:
        return f"Błąd podczas klasyfikacji: {e}", pd.DataFrame()



# chmura słów
def create_word_cloud(nearest_neighbors_df):
    text_for_wordcloud = " ".join(nearest_neighbors_df['overview'].dropna())

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate(text_for_wordcloud)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

    st.pyplot(fig)

# słupkowy
def create_bar_chart(nearest_neighbors_df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(nearest_neighbors_df['overview'])

    word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    word_freq = word_counts.sum().sort_values(ascending=False)

    top_10_words = word_freq.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))

    top_10_words.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)




