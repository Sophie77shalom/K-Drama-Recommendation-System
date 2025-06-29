# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the recommendation function
def hybrid_recommendations(title, cosine_sim, df, indices):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    drama_indices = [i[0] for i in sim_scores]
    return df[['Name', 'Genres', 'Rating', 'Popularity_Score']].iloc[drama_indices]

# Load the model
@st.cache_resource
def load_model():
    with open('kdrama_recommender.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Streamlit UI
st.title('K-Drama Recommendation System')
st.write("Discover your next favorite Korean drama!")
selected_drama = st.selectbox("Select a K-Drama:", model['df']['Name'].tolist())
if st.button("Get Recommendations"):
    recs = hybrid_recommendations(selected_drama, model['cosine_sim'], model['df'], model['indices'])
    st.write(recs)

# Sidebar for additional info
st.sidebar.header("About")
st.sidebar.info("This system recommends K-Dramas based on content similarity and popularity.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Python, Scikit-learn, and Streamlit")