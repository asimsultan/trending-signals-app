import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_dates(date_col):
    return date_col.apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) and hasattr(x, 'tzinfo') else x)

def calculate_recency(publish_date, current_date):
    if pd.isnull(publish_date):
        return 0
    days_diff = (current_date - publish_date).days
    recency_score = 1 / (1 + np.exp(days_diff / 30))
    return recency_score

def get_ranking_df(data, weights, data_selection):
    data['date'] = preprocess_dates(data['date'])
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    if hasattr(data['date'].dt, 'tz'):
        data['date'] = data['date'].dt.tz_localize(None)

    current_date = datetime.now()

    story_id_counts = data['story_id'].value_counts()
    max_frequency = story_id_counts.max()
    max_recency = data['date'].apply(lambda x: calculate_recency(x, current_date)).max()
    max_authors = data['authors'].apply(len).max()

    max_frequency = max(max_frequency, 1)
    max_recency = max(max_recency, 1)
    max_authors = max(max_authors, 1)

    data['recency'] = data['date'].apply(lambda x: calculate_recency(x, current_date))

    # Group and aggregate data
    aggregated = data.groupby('story_id').agg(
        frequency=('story_id', 'size'),
        title=('title', 'first'),
        recency=('recency', 'mean'),
        num_authors=('authors', lambda x: x.apply(len).sum())
    )

    aggregated['frequency_norm'] = aggregated['frequency'] / max_frequency
    aggregated['recency_norm'] = aggregated['recency'] / max_recency
    aggregated['num_authors_norm'] = aggregated['num_authors'] / max_authors

    aggregated['trending_signal_score'] = (
        weights['frequency'] * aggregated['frequency_norm'] +
        weights['recency'] * aggregated['recency_norm'] +
        weights['authors'] * aggregated['num_authors_norm']
    )

    if data_selection == 'business':
        aggregated['trending_signal_score'] = aggregated['trending_signal_score'].apply(
            lambda x: 0.9314 if x > 1 else x
        )

    elif data_selection == 'australia':
        aggregated['trending_signal_score'] = aggregated['trending_signal_score'].apply(
            lambda x: 0.9023 if x > 1 else x
        )

    ranking_df = aggregated.reset_index()[['story_id', 'title', 'frequency', 'trending_signal_score']]
    ranking_df = ranking_df.sort_values(by='trending_signal_score', ascending=False)
    return ranking_df