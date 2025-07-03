import streamlit as st
import pandas as pd
import pickle
import joblib
import os
import gdown

# Google Drive file ID
GDRIVE_FILE_ID = "1TxACzqnHy1wUrwsmR7Cgjq02J4vrs41w"
SIMILARITY_FILENAME = "similarity_compressed.pkl"

# Step 1: Download file if not already present
@st.cache_data
def download_similarity_file():
    if not os.path.exists(SIMILARITY_FILENAME):
        st.info("🔄 Downloading similarity matrix...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, SIMILARITY_FILENAME, quiet=False)
        st.success("✅ Download complete!")
    return joblib.load(SIMILARITY_FILENAME)

# Step 2: Load movies and similarity data
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = download_similarity_file()

# Step 3: Recommend movies
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movies_list]

# Step 4: Streamlit UI
st.title("🎬 Movie Recommender System")

selected_movie_name = st.selectbox(
    "Select a movie to get recommendations:",
    movies['title'].values
)

if st.button("Recommend"):
    st.subheader("Top 5 Recommendations:")
    for movie in recommend(selected_movie_name):
        st.write("•", movie)
