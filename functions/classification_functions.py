import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# tworzenie modelu knn
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


# tworzenie modelu svm
def create_svm_model(classification_df):
    classification_df = classification_df.copy()
    classification_df['text'] = classification_df['original_title'] + " " + classification_df['overview']
    X = classification_df['text']
    y = classification_df['vote_category']

    model = make_pipeline(TfidfVectorizer(), LinearSVC())
    model.fit(X, y)
    return model


# klasyfikacja dokumentów - knn
def classify_with_knn(model, title, overview, classification_df):
    text = title + " " + overview
    vector = model.named_steps['tfidfvectorizer'].transform([text])

    # Pobranie sąsiadów
    distances, indices = model.named_steps['kneighborsclassifier'].kneighbors(vector, return_distance=True)
    nearest_neighbors_df = classification_df.iloc[indices[0]].reset_index(drop=True)

    # Mapowanie kategorii
    category_mapping = {
        1: "bardzo zły",
        2: "zły",
        3: "średni",
        4: "dobry",
        5: "bardzo dobry"
    }
    nearest_neighbors_df['vote_category'] = pd.to_numeric(nearest_neighbors_df['vote_category'], errors='coerce')
    nearest_neighbors_df['vote_category'] = nearest_neighbors_df['vote_category'].map(category_mapping).fillna("Nieznana kategoria")

    # Predykcja na podstawie sąsiadów
    predicted_category = nearest_neighbors_df['vote_category'].mode()[0]
    return predicted_category, nearest_neighbors_df


# klasyfikacja dokumentów - svm
def classify_with_svm(model, title, overview):
    text = title + " " + overview
    vector = model.named_steps['tfidfvectorizer'].transform([text])

    # Predykcja SVM
    predicted_category = model.named_steps['linearsvc'].predict(vector)[0]

    # Debugowanie - wyświetlenie surowej wartości zwróconej przez SVM
    print(f"Predykcja zwrócona przez SVM (przed mapowaniem): {predicted_category}")
    print(f"Typ danych predykcji: {type(predicted_category)}")

    # Konwersja na int, jeśli jest to string
    if isinstance(predicted_category, str):
        predicted_category = int(predicted_category)

    # Mapowanie kategorii
    category_mapping = {
        1: "bardzo zły",
        2: "zły",
        3: "średni",
        4: "dobry",
        5: "bardzo dobry"
    }
    predicted_category = category_mapping.get(predicted_category, "Nieznana kategoria")
    return predicted_category