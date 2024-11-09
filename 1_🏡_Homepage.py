import streamlit as st
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import warnings
import openpyxl
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="✨"
)

root_container = st.container()
root_container.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
        max-width: 100%;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.success("Made by 👩🏻‍💻 Ditha Lozera Devi")

st.markdown(
    """
    <div style="background-color:#333; padding:15px;">
        <h1 style='text-align: center;'>
            <span style='display: block;font-size: 50px; color: white'>
                📈 <strong>Sentiment Analysis Access by KAI</strong> 🚇
            </span>
        </h1>
            <span style='display: block; font-size: medium; color: white; text-align: center;'>
                "Website ini menggunakan metode Word2Vec model Skip-gram dan algoritma SVM kernel RBF"
            </span>
    </div>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("Download Template Data")
    st.write("Klik tautan di bawah ini untuk mengunduh template data.")

    # Tampilkan tautan Google Drive
    show_drive_link()

def show_drive_link():
    st.markdown("[Link ke Google Drive](https://drive.google.com/drive/folders/1tX_lDEhBjcmIEuRgQ9EHF8YkwYN5kaTO?usp=sharing)")

if __name__ == "__main__":
    main()

st.title("Data File Reader")

uploaded_file = st.file_uploader("Upload your file here:", type=(["csv","txt","xlsx","xls"]))

st.warning("Warning: upload the data for give you the result!")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    st.write("Data from Excel file:")
    st.dataframe(df['review'], use_container_width=True)

    import re

    def remove_emoticons(text):
        emoticon_pattern = re.compile(u'['
                                    u'\U0001F600-\U0001F64F'
                                    u'\U0001F300-\U0001F5FF'
                                    u'\U0001F680-\U0001F6FF'
                                    u'\U0001F1E0-\U0001F1FF'
                                    ']+', flags=re.UNICODE)
        return emoticon_pattern.sub(r'', text)

    def remove_special_characters(text):
        special_char_pattern = re.compile(r'[^a-zA-Z\s]')
        return special_char_pattern.sub('', text)

    df['review'] = df['review'].apply(remove_emoticons)
    df['review'] = df['review'].apply(remove_special_characters)

    stop = stopwords.words('indonesian')
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def preprocess_text(text):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop]
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return stemmed_tokens

    df['review'] = df['review'].apply(lambda x: ' '.join(preprocess_text(x)))

        # Convert review column to list of words
    df['review'] = df['review'].apply(lambda x: word_tokenize(x))
    
    # Train Word2Vec model with skip-gram
    word2vec_model = Word2Vec(sentences=df['review'], vector_size=100, window=5, min_count=1, sg=1)

    # Convert words to vectors
    def vectorize_text(text):
        vector = np.zeros(100)
        count = 0
        for word in text:
            if word in word2vec_model.wv:
                vector += word2vec_model.wv[word]
                count += 1
        if count != 0:
            vector /= count
        return vector

    df['review_vector'] = df['review'].apply(vectorize_text)

    if 'sentiment' in df.columns:
        # Prepare data for SVM model
        X = np.array(df['review_vector'].tolist())
        y = df['sentiment']

        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)

        svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        svm_model.fit(X_smote, y_smote)

        df['predicted_sentiment'] = svm_model.predict(X)

        st.title("Analysis Result")
        st.dataframe(df[['review', 'predicted_sentiment']], use_container_width=True)

        df.to_csv("predicted_sentiments.csv", index=False)
        with open("predicted_sentiments.csv", "rb") as file:
            st.download_button(
                label="Download the result CSV",
                data=file,
                file_name="predicted_sentiments.csv",
                mime="text/csv"
            )
    else:
        st.error("The uploaded file must contain a 'sentiment' column.")

    # HTML for top bar
    top_bar = """
    <div style="background-color:#333; padding:20px">
        <h3 style="color:white; font-size: 40px; font-weight: bold;text-align:center;">
     ❤️‍🔥Thank you for visiting, hope you like it❤️‍🔥
     </h3>
    </div>
    """

    st.markdown(top_bar, unsafe_allow_html=True)
