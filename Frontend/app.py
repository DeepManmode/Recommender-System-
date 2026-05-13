import streamlit as st
import pickle
import pandas as pd
import requests

def recommand(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    # result = [(i, float(score)) for i, score in result]
    # movies_list=result
    recommended_movies = []
    for i in movies_list:
        movie_id =i[0]
        #fetch poster from api
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies
movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
movies =pd.DataFrame(movies_dict)

similarity = pickle.load(open("similarity.pkl", "rb"))
st.title('Movie Recommender system')

selected_movie_name= st.selectbox(
    "Select movie to recommend",
   movies['title'].values,
)

if st.button("Recommend"):
    recommendations =recommand(selected_movie_name)
    for i in recommendations:
        st.write(i)