from textblob import TextBlob
from rapidfuzz.distance import Levenshtein
import pandas as pd
import string


def preprocess_text(text):
    text = text.lower()  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    return text

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
