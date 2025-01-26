from sklearn.preprocessing import normalize
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# miara cosinusa
def cosine(tfidf_matrix, query_vector):
    tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2')
    query_vector_normalized = normalize(query_vector, norm='l2')

    cosine_similarity_scores = tfidf_matrix_normalized @ query_vector_normalized.T
    cosine_similarity_scores = cosine_similarity_scores.flatten()

    return cosine_similarity_scores


# miara Jaccarda
def jaccard(filtered_df, corrected_query): 
    count_vectorizer = CountVectorizer(binary=True, stop_words='english')
    binary_matrix = count_vectorizer.fit_transform(filtered_df['overview']).toarray()
    query_binary_vector = count_vectorizer.transform([corrected_query]).toarray()
    jaccard_scores = [
        jaccard_score(binary_matrix[i], query_binary_vector[0], average='binary')
        for i in range(binary_matrix.shape[0])
    ]

    return jaccard_scores


# LSI
def lsi(tfidf_matrix, query_vector):
    svd = TruncatedSVD(n_components=100, random_state=42)
    lsi_matrix = svd.fit_transform(tfidf_matrix)
    query_lsi_vector = svd.transform(query_vector)
    lsi_similarity_scores = cosine_similarity(lsi_matrix, query_lsi_vector)
    lsi_similarity_scores = lsi_similarity_scores.flatten()

    return lsi_similarity_scores