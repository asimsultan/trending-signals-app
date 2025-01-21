import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import math
import boto3
import pickle
import io, os

# Authentication logic
def login():
    st.sidebar.title("Login")

    username = st.sidebar.text_input("Username", key="username")
    password = st.sidebar.text_input("Password", type="password", key="password")

    if st.sidebar.button("Login"):
        name_key = os.getenv("USER_NAME")
        password_key = os.getenv("PASSWORD")

        if username == name_key and password == password_key:
            st.session_state["authenticated"] = True
            st.session_state["message"] = f"Welcome, {username}!"
        else:
            st.session_state["authenticated"] = False
            st.sidebar.error("Invalid credentials")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# Preprocess dates function
def preprocess_dates(date_col):
    return date_col.apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) and hasattr(x, 'tzinfo') else x)


# Recency calculation function
def calculate_recency(new_date, current_date):
    if pd.isnull(new_date):
        return 0
    days_diff = (current_date - new_date).days
    recency_score = 1 / (1 + np.exp(days_diff / 30))
    return recency_score


# Ranking data processing
def get_ranking_df(data, weights, data_selection):
    data['new_date'] = preprocess_dates(data['new_date'])
    current_date = datetime.now()

    max_frequency = data['story_id'].value_counts().max()
    max_recency = data['new_date'].apply(lambda x: calculate_recency(x, current_date)).max()
    max_authors = data['authors'].apply(len).max()
    max_page_rank = data['page_rank'].max()

    data['recency'] = data['new_date'].apply(lambda x: calculate_recency(x, current_date))
    aggregated = data.groupby('story_id').agg(
        frequency=('story_id', 'size'),
        title=('title', 'first'),
        recency=('recency', 'mean'),
        num_authors=('authors', lambda x: x.apply(len).sum()),
        page_rank=('page_rank', 'mean')
    )
    aggregated['frequency_norm'] = aggregated['frequency'] / max_frequency
    aggregated['recency_norm'] = aggregated['recency'] / max_recency
    aggregated['num_authors_norm'] = aggregated['num_authors'] / max_authors
    aggregated['page_rank_norm'] = aggregated['page_rank'] / max_page_rank

    aggregated['trending_signal_score'] = (
            weights['frequency'] * aggregated['frequency_norm'] +
            weights['recency'] * aggregated['recency_norm'] +
            weights['authors'] * aggregated['num_authors_norm'] +
            weights['page_rank'] * aggregated['page_rank_norm']
    )

    ranking_df = aggregated.reset_index()[['story_id', 'title', 'frequency', 'trending_signal_score']]
    ranking_df = ranking_df.sort_values(by='trending_signal_score', ascending=False)
    return ranking_df


# Read data from S3
def read_pkl_from_s3(bucket_name, object_name):
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        file_content = response['Body'].read()
        data = pickle.load(io.BytesIO(file_content))
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


# Cache data loading
@st.cache_data
def load_data(selection):
    bucket_name = "trending-signal-bucket"
    file_name = 'Business_df.pkl' if selection == "Business" else 'Australia_df.pkl'

    data = read_pkl_from_s3(bucket_name, file_name)
    # data = pd.read_pickle(file_name)
    return data


# App starts here for authenticated users
st.title("Trending Signals: Dynamic Weight Adjustment")
st.sidebar.header("Data Selection")
data_selection = st.sidebar.selectbox("Select Data Type", options=["Business", "Australia"], index=0)
data = load_data(data_selection)

st.sidebar.header("Adjust Weights")
frequency_weight = st.sidebar.slider("Frequency Weight", 0.0, 1.0, 0.4, 0.1)
page_rank_weight = st.sidebar.slider("Page Rank", 0.0, 1.0, 0.3, 0.1)
recency_weight = st.sidebar.slider("Recency Weight", 0.0, 1.0, 0.2, 0.1)
authors_weight = st.sidebar.slider("Authors Weight", 0.0, 1.0, 0.1, 0.1)

total_weight = frequency_weight + page_rank_weight + recency_weight + authors_weight
if not math.isclose(total_weight, 1.0, rel_tol=1e-6):
    st.warning("The total weight must sum to exactly 1. Adjust the sliders accordingly.")
    st.stop()

weights = {
    'frequency': frequency_weight,
    'page_rank': page_rank_weight,
    'recency': recency_weight,
    'authors': authors_weight
}

ranking_df = get_ranking_df(data, weights, data_selection)
st.subheader("Top Trending Stories")
st.dataframe(ranking_df)