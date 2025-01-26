from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

import functions.text_functions as txt


# chmura słów
def create_word_cloud(nearest_neighbors_df):
    text_for_wordcloud = " ".join(
        nearest_neighbors_df['overview']
        .dropna()
        .apply(txt.preprocess_text)  
    )

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Wistia'
    ).generate(text_for_wordcloud)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

    st.pyplot(fig)


# wykres słupkowy
def create_bar_chart(nearest_neighbors_df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(nearest_neighbors_df['overview'])

    word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    word_freq = word_counts.sum().sort_values(ascending=False)

    top_10_words = word_freq.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))

    top_10_words.plot(kind='bar', color=(0.96, 0.77, 0.09), ax=ax)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)