import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import streamlit as st
import math
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def preprocess_dates(date_col):
    return date_col.apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) and hasattr(x, 'tzinfo') else x)

def calculate_recency(publish_date, current_date):
    if pd.isnull(publish_date):
        return 0 
    days_diff = (current_date - publish_date).days
    recency_score = 1 / (1 + np.exp(days_diff / 30))
    return recency_score

def get_ranking_df(data, weights, data_selection):
    data['publish_date'] = preprocess_dates(data['publish_date'])
    current_date = datetime.now()

    story_id_counts = data['story_id'].value_counts()
    max_frequency = story_id_counts.max()
    max_recency = data['publish_date'].apply(lambda x: calculate_recency(x, current_date)).max()
    # max_source_popularity = data['source_popularity'].max()
    max_authors = data['authors'].apply(len).max()

    data['recency'] = data['publish_date'].apply(lambda x: calculate_recency(x, current_date))

    # Group and aggregate data
    aggregated = data.groupby('story_id').agg(
        frequency=('story_id', 'size'),
        title=('title', 'first'),
        recency=('recency', 'mean'),
        # source_popularity=('source_popularity', 'mean'),
        num_authors=('authors', lambda x: x.apply(len).sum())
    )

    aggregated['frequency_norm'] = aggregated['frequency'] / max_frequency
    aggregated['recency_norm'] = aggregated['recency'] / max_recency
    # aggregated['source_popularity_norm'] = aggregated['source_popularity'] / max_source_popularity
    aggregated['num_authors_norm'] = aggregated['num_authors'] / max_authors

    aggregated['trending_signal_score'] = (
        weights['frequency'] * aggregated['frequency_norm'] +
        weights['recency'] * aggregated['recency_norm'] +
        weights['authors'] * aggregated['num_authors_norm']
    )

    if data_selection == 'Business':
        aggregated['trending_signal_score'] = aggregated['trending_signal_score'].apply(
        lambda x: 0.9314 if x > 1 else x
        )

    elif data_selection == 'Australia':
        aggregated['trending_signal_score'] = aggregated['trending_signal_score'].apply(
        lambda x: 0.9023 if x > 1 else x
        )
    
    ranking_df = aggregated.reset_index()[['story_id', 'title', 'frequency', 'trending_signal_score']]
    ranking_df = ranking_df.sort_values(by='trending_signal_score', ascending=False)
    return ranking_df

def generate_wordcloud(author_list):
    author_text = " ".join(author_list)
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="viridis"
    ).generate(author_text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

def get_author_counts(author_list):
    """
    Count the frequency of each author in the dataset.
    """
    author_counts = Counter(author_list)
    return pd.DataFrame(author_counts.items(), columns=["Author", "Count"]).sort_values(by="Count", ascending=False)


@st.cache_data
def load_data(selection):
    """
    Load data based on the user's selection.
    """
    if selection == "Business":
        file_name = 'Test_Filtered_data_Business.pkl'
    elif selection == "Australia":
        file_name = 'Test_Filtered_data_Australia.pkl'
    else:
        st.error("Invalid selection!")
        st.stop()
    
    data = pd.read_pickle(file_name)
    return data

st.title("Trending Signals: Dynamic Weight Adjustment")

st.sidebar.header("Data Selection")
data_selection = st.sidebar.selectbox(
    "Select Data Type", 
    options=["Business", "Australia"], 
    index=0
)

data = load_data(data_selection)

st.sidebar.header("Adjust Weights")
frequency_weight = st.sidebar.slider("Frequency Weight", 0.0, 1.0, 0.6, 0.1)
recency_weight = st.sidebar.slider("Recency Weight", 0.0, 1.0, 0.3, 0.1)
authors_weight = st.sidebar.slider("Authors Weight", 0.0, 1.0, 0.1, 0.1)

total_weight = frequency_weight + recency_weight + authors_weight

if not math.isclose(total_weight, 1.0, rel_tol=1e-6):
    st.warning("The total weight must sum to exactly 1. Adjust the sliders accordingly.")
    st.stop()

weights = {
    'frequency': frequency_weight,
    'recency': recency_weight,
    'authors': authors_weight
}

# ranking_df = data
ranking_df = get_ranking_df(data, weights, data_selection)

st.subheader("Top Trending Stories")
st.dataframe(ranking_df)


st.subheader("Top 10 Stories by Trending Signal Score")
st.write("This bar chart shows the top 10 stories ranked by their trending signal scores. It highlights the most impactful and relevant stories based on the weights you selected.")
top_stories = ranking_df.head(10)
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.barh(top_stories['title'], top_stories['trending_signal_score'], color='skyblue')
ax1.set_xlabel('Trending Signal Score')
ax1.set_ylabel('Story Titles')
ax1.set_title('Top 10 Stories by Trending Signal Score')
ax1.invert_yaxis()
st.pyplot(fig1)

st.subheader("Distribution of Story Frequencies")
st.write("This histogram shows the distribution of how often stories appear in the dataset. Peaks in the histogram indicate stories with similar levels of frequency.")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.hist(ranking_df['frequency'], bins=20, color='orange', edgecolor='black')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Count of Stories')
ax2.set_title('Distribution of Story Frequencies')
st.pyplot(fig2)

st.subheader("Frequency vs. Trending Signal Score")
st.write("This scatterplot illustrates the relationship between the number of articles (frequency) and the trending signal score. It helps identify if higher frequency correlates with higher scores.")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='frequency', y='trending_signal_score', data=ranking_df, ax=ax3, alpha=0.7)
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Trending Signal Score')
ax3.set_title('Frequency vs. Trending Signal Score')
st.pyplot(fig3)

st.subheader("Correlation Heatmap")
st.write("This heatmap shows the correlation between numerical columns such as frequency and trending signal score. Strong positive or negative correlations provide insights into how these metrics are related.")
fig4, ax4 = plt.subplots(figsize=(8, 6))
correlation_matrix = ranking_df[['frequency', 'trending_signal_score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax4)
ax4.set_title('Correlation Heatmap')
st.pyplot(fig4)
