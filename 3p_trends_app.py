import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import numpy as np
from datetime import timedelta
import math
import boto3
import pickle
import io, os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import time
# import reddit_report

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
            # Initialize page state after successful login
            if "current_page" not in st.session_state:
                st.session_state["current_page"] = None
        else:
            st.session_state["authenticated"] = False
            st.sidebar.error("Invalid credentials")

import base64  # For encoding the PDF


def show_navigation():
    st.sidebar.header("Navigation")
    if st.sidebar.button("Trending Scores"):
        st.session_state["current_page"] = "trending"
        st.rerun()
    if st.sidebar.button("Aggregator Scores"):
        st.session_state["current_page"] = "aggregator"
        st.rerun()
    if st.sidebar.button("Reddit Signals"):
        st.session_state["current_page"] = "reddit"
        st.rerun()
    if st.sidebar.button("Overall View"):
        st.session_state["current_page"] = "overall"
        st.rerun()
    if st.sidebar.button("Export Report"):
        progress_bar = st.progress(0, "Generating Report...")
        try:
            # Simulate report generation time (replace with actual report generation)
            for i in range(10):  # Adjust the range to reflect your actual report generation steps
                time.sleep(0.5)  # Replace with actual report generation time
                progress_bar.progress((i + 1) * 10, "Generating Report...")

            # reddit_report.generate_report()  # Generate the report

            progress_bar.empty()  # Remove the progress bar

            with open("business_trending_report.pdf", "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.download_button(
                label="Download Report",
                data=pdf_bytes,
                file_name="business_trending_report.pdf",
                mime="application/pdf"
            )

            st.success("Report generated and ready for download!")

        except FileNotFoundError:
            st.error(
                "Report file 'business_trending_report.pdf' not found. Please check if the report was generated successfully.")
            progress_bar.empty()
        except Exception as e:
            st.error(f"An error occurred during report generation or download: {e}")
            progress_bar.empty()

def preprocess_dates(date_col):
    return date_col.apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) and hasattr(x, 'tzinfo') else x)

def calculate_recency(new_date, current_date):
    if pd.isnull(new_date):
        return 0
    days_diff = (current_date - new_date).days
    recency_score = 1 / (1 + np.exp(days_diff / 30))
    return recency_score

def get_ranking_df(data, weights, data_selection):
    data['Story Date'] = pd.to_datetime(data['Story Date'])

    data['new_date'] = preprocess_dates(data['new_date'])
    data['Story Date'] = preprocess_dates(data['Story Date'])
    # data['new_date'] = pd.to_datetime(data['new_date'], errors='coerce')

    current_date = pd.Timestamp.now()
    max_frequency = data['story_id'].value_counts().max()
    # max_recency = data['new_date'].apply(lambda x: calculate_recency(x, current_date)).max()
    max_recency = data['Story Date'].apply(lambda x: calculate_recency(x, current_date)).max()

    max_authors = data['authors'].apply(len).max()
    max_page_rank = data['page_rank'].max()

    max_frequency = max(max_frequency, 1)
    max_recency = max(max_recency, 1)
    max_authors = max(max_authors, 1)
    max_page_rank = max(max_page_rank, 1)

    # data['recency'] = data['new_date'].apply(lambda x: calculate_recency(x, current_date))
    data['recency'] = data['Story Date'].apply(lambda x: calculate_recency(x, current_date))

    aggregated = data.groupby('story_id').agg(
        frequency=('story_id', 'size'),
        title=('title', 'first'),
        recency=('recency', 'mean'),
        date=('new_date', 'first'),
        num_authors=('authors', lambda x: x.apply(len).sum()),
        page_rank=('page_rank', 'mean'),
        views=('views', 'first'),
        internal_rank=('rank', 'first'),
        rank_mask=('rank_mask', 'first'),
        top_designations=('top_designations', 'first'),
        mediaCount=('mediaCount', 'first'),
        sourceCount=('sourceCount', 'first'),
        storyDate=('Story Date', 'first')
    )

    aggregated['frequency_norm'] = aggregated['frequency'] / max_frequency

    aggregated['recency_norm'] = aggregated['recency'] / max_recency
    # aggregated['source_popularity_norm'] = aggregated['source_popularity'] / max_source_popularity
    aggregated['num_authors_norm'] = aggregated['num_authors'] / max_authors
    aggregated['page_rank_norm'] = aggregated['page_rank'] / max_page_rank

    aggregated_frequency = weights['frequency'] * aggregated['frequency_norm']
    aggregated_page_rank = weights['page_rank'] * aggregated['page_rank_norm']
    aggregated_recency = weights['recency'] * aggregated['recency_norm']
    aggregated_authors = weights['authors'] * aggregated['num_authors_norm']

    aggregated['trending_signal_score'] = (
            aggregated_frequency + aggregated_recency + aggregated_authors + aggregated_page_rank)

    ranking_df = aggregated.reset_index()[
        ['story_id', 'title', 'frequency', 'trending_signal_score', 'page_rank_norm', 'views', 'internal_rank',
         'rank_mask', 'top_designations', 'mediaCount', 'sourceCount', 'date', 'storyDate']]
    ranking_df = ranking_df.sort_values(by='trending_signal_score', ascending=False)
    # ranking_df['story_id'] = ranking_df['story_id'].apply(
    #     lambda x: f'<a href="https://ground.news/article/{x}" target="_blank">{x}</a>')

    ranking_df['link'] = ranking_df['story_id'].apply(
        lambda x: f'https://ground.news/article/{x}')
    return ranking_df


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


def get_latest_available_data(selection, category, apply_themes):
    max_days_to_check = 7  # Number of past days to check if today's data isn't available
    bucket_name = "trending-signal-bucket"  # Correct bucket name without the date

    for i in range(max_days_to_check):
        check_date = (pd.Timestamp.now() - timedelta(days=i)).strftime("%Y-%m-%d")

        if category == 'trending':
            if apply_themes==False:
                object_key = f"{check_date}/Business_df_unfiltered.pkl" if selection == "Business" else f"{check_date}/Australia_df_unfiltered.pkl"
            else:
                object_key = f"{check_date}/Business_df.pkl" if selection == "Business" else f"{check_date}/Australia_df.pkl"
            try:
                data = read_pkl_from_s3(bucket_name, object_key)
                if data is not None:
                    return data, check_date
            except Exception as e:
                continue
        elif category == 'aggregator':
            object_key = f"{check_date}/Aggregated_results.pkl"
            try:
                data = read_pkl_from_s3(bucket_name, object_key)
                if data is not None:
                    return data, check_date
            except Exception as e:
                continue
        elif category == 'reddit':

            if apply_themes==False:
                object_key = f"{check_date}/Reddit_signals_unfiltered.pkl"
            else:
                object_key = f"{check_date}/Reddit_signals_filtered.pkl"
            try:
                data = read_pkl_from_s3(bucket_name, object_key)
                # data['date'] = pd.to_datetime(data['createdAt'])
                # one_day_ago = pd.Timestamp.utcnow() - pd.Timedelta(hours=36)
                # data = data[data['date'] >= one_day_ago]
                # data = data[['storyId', 'title', 'downvotes', 'upvotes', 'createdAt', 'url', 'storyUrl', 'redditLink',
                #              'velocity', 'compositeScore', 'normalizedScore', 'subredditRoute', 'subreddit_frequency']]

                # first_columns = ["storyId", "title", "compositeScore", "rank_mask", "subreddit_count"]
                # data = data[first_columns + [col for col in data.columns if col not in first_columns]]

                first_columns = ["storyId", "title", "compositeScore", "rank_mask", "subreddit_count"]
                remaining_columns = [col for col in data.columns if col not in first_columns]

                print('Remaining Columns', remaining_columns)
                data = data[first_columns + remaining_columns]
                data = data.drop(['story_id'], axis=1)

                # data.rename(columns={'sourceCount_x': 'sourceCount'}, inplace=True)

                print('Remaining Columns now', data.columns)

                if data is not None:
                    return data, check_date
            except Exception as e:
                print(f"Error for {object_key}: {e}")
                continue

    print("No data available for the past 7 days.")
    return None, None


@st.cache_data
def load_data(selection, category, apply_themes):
    data, the_date = get_latest_available_data(selection, category, apply_themes)
    return data


def setup_grid_options(df):
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_column(
        "story_id",
        header_name="Story ID",
        width=150,
        maxWidth=180,
        filter=True,  # Enable filtering
        filterParams={"filter": "agTextColumnFilter"},  # Use text-based filter
        floatingFilter=True,  # Show search box under the column header
    )

    gb.configure_column(
        "title",
        header_name="Title",
        width=180,
        maxWidth=380,
        filter=True,  # Enable filtering
        filterParams={"filter": "agTextColumnFilter"},  # Use text-based filter
        floatingFilter=True,  # Show search box under the column header
    )

    gb.configure_column(
        "top_designations",
        header_name="Top Designations",
        width=120,
        maxWidth=170,
        filter=True,  # Enable filtering
        filterParams={"filter": "agTextColumnFilter"},  # Use text-based filter
        floatingFilter=True,  # Show search box under the column header
    )

    gb.configure_column(
        "rank_mask",
        header_name="Rank",
        width=150,
        maxWidth=250,
        filter=True,  # Enable filtering
        filterParams={"filter": "agTextColumnFilter"},  # Use text-based filter
        floatingFilter=True,  # Show search box under the column header
    )

    gb.configure_column("mediaCount", header_name="Media Count", width=130, maxWidth=150)
    gb.configure_column("sourceCount", header_name="Source Count", width=130, maxWidth=150)
    gb.configure_column("frequency", header_name="Frequency", width=40, maxWidth=120)
    gb.configure_column("trending_signal_score", header_name="Signal Score", width=150, maxWidth=120)
    gb.configure_column("page_rank_norm", header_name="Page Rank", width=100, maxWidth=120)
    gb.configure_column("views", header_name="Views", width=60, maxWidth=90)
    gb.configure_column("date", header_name="Date", width=200, maxWidth=100)
    gb.configure_column("internal_rank", header_name="Internal_rank", width=100, maxWidth=120)

    # Configure clickable link column with strict width
    gb.configure_column(
        "link",
        header_name="Story Link",
        width=80,
        maxWidth=220,
        cellRenderer="""
            function(params) {
                return `<a href="${params.value}" target="_blank" style="text-decoration: none;">
                            <span style="color: blue; font-size: 16px;">🔗</span>
                        </a>`;
            }
        """  # Render a blue link icon (Unicode 🔗)
    )

    # Build grid options with strict layout and disable auto column sizing
    grid_options = gb.build()

    grid_options.update({
        "domLayout": "autoHeight",
        "animateRows": True,
        "suppressRowVirtualisation": True,
        "rowBuffer": 10,
        "cellFlashDuration": 700,
        "cellFadeDuration": 1000,
        "ensureDomOrder": True,
        "suppressMaxRenderedRowRestriction": True,
        "suppressAutoSize": True,
        "defaultColDef": {"suppressSizeToFit": True},
    })
    return grid_options

def show_trending_scores():
    st.title("Trending Signals: Dynamic Weight Adjustment")
    apply_themes = st.sidebar.toggle("Apply Themes/Interests", value=False)
    current_date = pd.Timestamp.now().strftime("%b %d, %Y")
    st.markdown(
        f"<div style='text-align: right; font-size: 14px; font-weight: bold;'>The given data is fetched on {current_date}</div>",
        unsafe_allow_html=True)

    st.sidebar.header("Data Selection")
    data_selection = st.sidebar.selectbox("Select Data Type", options=["Business", "Australia"], index=0)

    data = load_data(data_selection, category='trending', apply_themes=apply_themes)

    st.sidebar.header("Adjust Weights")
    frequency_weight = st.sidebar.slider("Frequency Weight", 0.0, 1.0, 0.4, 0.1)
    page_rank_weight = st.sidebar.slider("Page Rank", 0.0, 1.0, 0.3, 0.1)
    recency_weight = st.sidebar.slider("Recency Weight", 0.0, 1.0, 0.2, 0.1)
    authors_weight = st.sidebar.slider("Authors Weight", 0.0, 1.0, 0.1, 0.1)

    total_weight = frequency_weight + page_rank_weight + recency_weight + authors_weight
    if not math.isclose(total_weight, 1.0, rel_tol=1e-6):
        st.warning("The total weight must sum to exactly 1. Adjust the sliders accordingly.")
        return

    weights = {
        'frequency': frequency_weight,
        'page_rank': page_rank_weight,
        'recency': recency_weight,
        'authors': authors_weight
    }

    ranking_df = get_ranking_df(data, weights, data_selection)
    grid_options = setup_grid_options(ranking_df)

    # Display which data version is being used
    theme_status = "with" if apply_themes else "without"
    st.subheader(f"Top Trending Stories ({theme_status} Themes/Interests)")

    custom_css = """
        <style>
        .ag-theme-streamlit {
            width: 100%; /* Expand table to full width */
        \    max-width: 1500px; /* Increase the maximum width for the table */
            min-width: 1200px; /* Set a minimum width for better visibility */
            margin: auto; /* Center the table on the page */
            overflow: auto; /* Allow scrolling if necessary */
        }
        </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    AgGrid(
        ranking_df,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        height=600,
        use_container_width=False,
    )

    if st.button('Export to Google Sheets'):
        try:
            url = export_to_gsheet(ranking_df, "Trending Signals Data")  # Pass ranking_df here
            st.success(f'Data exported successfully! [Open Sheet]({url})')
        except Exception as e:
            st.error(f'Error exporting data: {str(e)}')


def format_rank_per_story(value):
    if isinstance(value, dict):
        return ", ".join([f"{key}: {', '.join(map(str, val))}" for key, val in value.items()])
    return str(value)  # Convert any other type to string


def format_repeated_mentions(value):
    if isinstance(value, dict):
        return ", ".join([f"{key}: {val}" for key, val in value.items()])
    return str(value)  # Convert other types to string


def show_aggregator_scores():
    st.title("Aggregator Scores")

    current_date = pd.Timestamp.now().strftime("%b %d, %Y")
    st.markdown(
        f"<div style='text-align: right; font-size: 14px; font-weight: bold;'>The given data is fetched on {current_date}</div>",
        unsafe_allow_html=True)

    # Add weight sliders in sidebar
    st.sidebar.header("Adjust Aggregator Weights")
    agg_counts_weight = st.sidebar.slider("Aggregator Counts Weight", 0.0, 1.0, 0.4, 0.1)
    rank_story_weight = st.sidebar.slider("Rank per Story Weight", 0.0, 1.0, 0.3, 0.1)
    repeated_mentions_weight = st.sidebar.slider("Repeated Mentions Weight", 0.0, 1.0, 0.2, 0.1)
    recency_weight = st.sidebar.slider("Date Recency Weight", 0.0, 1.0, 0.1, 0.1)

    # Validate weights sum to 1
    total_weight = agg_counts_weight + rank_story_weight + repeated_mentions_weight + recency_weight
    if not math.isclose(total_weight, 1.0, rel_tol=1e-6):
        st.warning("The total weight must sum to exactly 1. Adjust the sliders accordingly.")
        return

    aggregator_data = load_data('', category='aggregator', apply_themes=True)


    aggregator_data["rank_per_story"] = aggregator_data["rank_per_story"].apply(format_rank_per_story)
    aggregator_data["repeated_mentions"] = aggregator_data["repeated_mentions"].apply(format_repeated_mentions)

    # Create grid options builder
    gb = GridOptionsBuilder.from_dataframe(aggregator_data)

    # Configure columns
    gb.configure_column("story_id", header_name="Story ID", width=150)
    gb.configure_column("title", header_name="Title", width=250)
    gb.configure_column("aggregator_score", header_name="Aggregator Score", width=150)
    gb.configure_column("aggregator_counts", header_name="Aggregator Counts", width=200)
    gb.configure_column("rank_per_story", header_name="Rank per Story", width=200)
    gb.configure_column("repeated_mentions", header_name="Repeated Mentions", width=200)
    gb.configure_column("top_news", header_name="Top News", width=100)
    gb.configure_column("views", header_name="Views", width=100)
    gb.configure_column("rank", header_name="Rank", width=100)
    gb.configure_column("rank_mask", header_name="Rank Mask", width=150)
    gb.configure_column("top_designations", header_name="Top Designations", width=150)
    gb.configure_column("mediaCount", header_name="Media Count", width=120)
    gb.configure_column("sourceCount", header_name="Source Count", width=120)

    # Configure grid options
    grid_options = gb.build()
    grid_options.update({
        "domLayout": "autoHeight",
        "animateRows": True,
        "suppressRowVirtualisation": True,
        "rowBuffer": 10,
        "cellFlashDuration": 700,
        "cellFadeDuration": 1000,
        "ensureDomOrder": True,
        "suppressMaxRenderedRowRestriction": True,
        "suppressAutoSize": True,
        "defaultColDef": {
            "suppressSizeToFit": True,
            "filter": True,
            "filterParams": {"filter": "agTextColumnFilter"},
            "floatingFilter": True
        }
    })

    # Custom CSS
    st.markdown("""
        <style>
        .ag-theme-streamlit {
            width: 100%;
            max-width: 1500px;
            min-width: 1200px;
            margin: auto;
            overflow: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Aggregator Results")

    AgGrid(
        aggregator_data,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        height=600,
        use_container_width=False
    )

    if st.button('Export to Google Sheets'):
        try:
            url = export_to_gsheet(aggregator_data, "Aggregator Scores Data")  # Pass data here
            st.success(f'Data exported successfully! [Open Sheet]({url})')
        except Exception as e:
            st.error(f'Error exporting data: {str(e)}')


def show_overall_view():
    st.title("Overall Story Analysis")

    current_date = datetime.now().strftime("%b %d, %Y")
    st.markdown(
        f"<div style='text-align: right; font-size: 14px; font-weight: bold;'>Data fetched on {current_date}</div>",
        unsafe_allow_html=True
    )

    data_selection = "Business"  # Default to Business data
    trending_data = load_data(data_selection, category='trending', apply_themes=False)
    # trending_data = pd.read_pickle('../Trending_Signals/Business_df.pkl')

    # bucket_name = "trending-signal-bucket"
    # object_name = '2025-02-19/Aggregated_results_feb19.pkl'
    # aggregator_data = read_pkl_from_s3(bucket_name, object_name)

    aggregator_data = load_data(data_selection, category='aggregator', apply_themes=False)
    reddit_data = load_data('', category='reddit', apply_themes=False )
    # aggregator_data = pd.read_pickle('../Trending_Signals/trending_signals_ingested/Aggregated_results_feb19.pkl')

    weights = {
        'frequency': 0.4,
        'page_rank': 0.3,
        'recency': 0.2,
        'authors': 0.1
    }
    trending_df = get_ranking_df(trending_data, weights, data_selection)

    trending_story_ids = set(trending_df['story_id'])
    aggregator_story_ids = set(aggregator_data['story_id'])
    reddit_story_ids = set(reddit_data['storyId'])

    common_story_ids = trending_story_ids.intersection(aggregator_story_ids)
    common_story_ids = common_story_ids.intersection(reddit_story_ids)

    trending_df_filtered = trending_df[trending_df['story_id'].isin(common_story_ids)]
    aggregator_data_filtered = aggregator_data[aggregator_data['story_id'].isin(common_story_ids)]
    reddit_data_filtered = reddit_data[reddit_data['storyId'].isin(common_story_ids)]

    total_trending = len(trending_story_ids)
    total_aggregator = len(aggregator_story_ids)
    total_reddit = len(reddit_story_ids)
    total_common = len(common_story_ids)

    st.sidebar.markdown("### Data Overview")
    st.sidebar.markdown(f"Total stories in Trending: {total_trending}")
    st.sidebar.markdown(f"Total stories in Aggregator: {total_aggregator}")
    st.sidebar.markdown(f"Total stories in Reddit: {total_reddit}")
    st.sidebar.markdown(f"Stories present in All: {total_common}")

    overall_df = trending_df_filtered.merge(aggregator_data_filtered, on='story_id', how='inner')
    overall_df = overall_df.merge(reddit_data_filtered, left_on='story_id', right_on='storyId', how='inner')
    overall_df.drop_duplicates(subset=['storyId'], inplace=True)


    overall_df = overall_df[
        ['storyId', 'title_x', 'trending_signal_score', 'aggregator_score', 'compositeScore', 'internal_rank',
         'rank_mask_x', 'date_x', 'link_x', 'upvotes', 'downvotes', 'createdAt', 'url', 'storyUrl', 'redditLink',
         'velocity', 'normalizedScore']]
    overall_df = overall_df.rename(
        columns={"title_x": "title", "internal_rank": "legacy_score", "rank_mask_x": "rank_mask", "date_x": "date",
                 "link_x": "link", "compositeScore": "reddit_signal_score"})
    # overall_df = overall_df[['story_id', 'title_x', 'trending_signal_score', 'aggregator_score', 'internal_rank', 'rank_mask_x', 'date_x', 'link_x']]
    # overall_df = overall_df.rename(
    #     columns={"title_x": "title", "internal_rank": "legacy_score", "rank_mask_x": "rank_mask", "date_x": "date", "link_x": "link"})
    gb = GridOptionsBuilder.from_dataframe(overall_df)

    gb.configure_column("story_id", width=150)
    gb.configure_column("title", header_name="Story Title", width=300)
    gb.configure_column("trending_signal_score", width=150)
    gb.configure_column("legacy_score", width=150)
    gb.configure_column("aggregator_score", width=150)
    gb.configure_column("date", width=150)

    gb.configure_column(
        "link",
        header_name="Story Link",
        width=80,
        cellRenderer="""
            function(params) {
                return `<a href="${params.value}" target="_blank" style="text-decoration: none;">
                            <span style="color: blue; font-size: 16px;">🔗</span>
                        </a>`;
            }
        """
    )

    grid_options = gb.build()
    grid_options.update({
        "domLayout": "autoHeight",
        "animateRows": True,
        "suppressRowVirtualisation": True,
        "rowBuffer": 10,
        "cellFlashDuration": 700,
        "cellFadeDuration": 1000,
        "ensureDomOrder": True,
        "suppressMaxRenderedRowRestriction": True,
        "suppressAutoSize": True,
        "defaultColDef": {
            "suppressSizeToFit": True,
            "filter": True,
            "filterParams": {"filter": "agTextColumnFilter"},
            "floatingFilter": True
        }
    })

    st.markdown("""
        <style>
        .ag-theme-streamlit {
            width: 100%;
            max-width: 1500px;
            min-width: 1200px;
            margin: auto;
            overflow: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    AgGrid(
        overall_df,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        height=600,
        use_container_width=False
    )

    if st.button('Export to Google Sheets'):
        try:
            url = export_to_gsheet(overall_df, "Aggregator Scores Data")  # Pass data here
            st.success(f'Data exported successfully! [Open Sheet]({url})')
        except Exception as e:
            st.error(f'Error exporting data: {str(e)}')


def export_to_gsheet(df, sheet_name="Trending Signals Data"):
    # Convert DataFrame to handle Timestamp objects
    df = df.copy()

    # Convert all datetime columns to string format
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        elif df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    # Convert any remaining non-string values to strings
    df = df.astype(str)

    # Define the scope
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets'
    ]

    try:
        # Create credentials dict from Streamlit secrets
        credentials_dict = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"],
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
        }

        # Authenticate using the credentials dictionary
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
        client = gspread.authorize(credentials)

        # Create a timestamp-based unique sheet name
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        unique_sheet_name = f"{sheet_name}_{timestamp}"

        # Create a new spreadsheet
        spreadsheet = client.create(unique_sheet_name)
        sheet = spreadsheet.sheet1

        # Share with specific emails
        emails_to_share = [
            'asim@ground.news',
            'asimsultan2@gmail.com',
            'haris@ground.news',
            'chuck@ground.news'
        ]

        # Share with each email
        for email in emails_to_share:
            spreadsheet.share(
                email,
                perm_type='user',
                role='writer',
                notify=True  # Send email notification
            )

        # Convert dataframe to list of lists
        data = [df.columns.values.tolist()] + df.values.tolist()

        # Update the sheet with data
        sheet.clear()
        sheet.update(data)

        # Get the spreadsheet URL
        sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
        return sheet_url

    except Exception as e:
        raise Exception(f"Error in Google Sheets export: {str(e)}")

def show_reddit_signals():
    st.title("Reddit Signals Analysis")
    apply_themes = st.sidebar.toggle("Apply Themes/Interests", value=False)
    current_date = datetime.now().strftime("%b %d, %Y")
    st.markdown(
        f"<div style='text-align: right; font-size: 14px; font-weight: bold;'>Data fetched on {current_date}</div>",
        unsafe_allow_html=True
    )

    # Load Reddit data using existing function
    # data = load_data('', category='reddit', apply_themes=False)
    data = load_data('', category='reddit', apply_themes=apply_themes)
    # data.drop_duplicates(subset=['storyId'], inplace=True)

    # data = data.sort_values(by='compositeScore', ascending=False)

    if data is None:
        st.error("Unable to load Reddit signals data")
        return

    # Create grid options
    gb = GridOptionsBuilder.from_dataframe(data)

    # Configure columns based on your data structure
    for column in data.columns:
        gb.configure_column(
            column,
            header_name=column.replace('_', ' ').title(),
            filter=True,
            filterParams={"filter": "agTextColumnFilter"},
            floatingFilter=True
        )

    theme_status = "with" if apply_themes else "without"
    st.subheader(f"Top Trending Stories ({theme_status} Themes/Interests)")

    grid_options = gb.build()
    grid_options.update({
        "domLayout": "autoHeight",
        "animateRows": True,
        "suppressRowVirtualisation": True,
        "rowBuffer": 10,
        "cellFlashDuration": 700,
        "cellFadeDuration": 1000,
        "ensureDomOrder": True,
        "suppressMaxRenderedRowRestriction": True,
        "suppressAutoSize": True,
        "defaultColDef": {"suppressSizeToFit": True}
    })

    AgGrid(
        data,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        height=600,
        use_container_width=False
    )

    if st.button('Export to Google Sheets'):
        try:
            url = export_to_gsheet(data, "Reddit Signals Data")
            st.success(f'Data exported successfully! [Open Sheet]({url})')
        except Exception as e:
            st.error(f'Error exporting data: {str(e)}')

def show_trending_scores():
    st.title("Trending Signals: Dynamic Weight Adjustment")
    apply_themes = st.sidebar.toggle("Apply Themes/Interests", value=False)
    current_date = pd.Timestamp.now().strftime("%b %d, %Y")
    st.markdown(
        f"<div style='text-align: right; font-size: 14px; font-weight: bold;'>The given data is fetched on {current_date}</div>",
        unsafe_allow_html=True)

    st.sidebar.header("Data Selection")
    data_selection = st.sidebar.selectbox("Select Data Type", options=["Business", "Australia"], index=0)

    data = load_data(data_selection, category='trending', apply_themes=apply_themes)

    if data.shape[0] == 0:
        st.subheader("Top Trending Stories")
        st.write("No trending signals in the last 24 hours.")  # Display the message
        return

    if 'new_date' not in data.columns:
        data.rename(columns={'date': 'new_date'}, inplace=True)
        data['new_date'] = pd.to_datetime(data['new_date'], utc=True)

    if 'rank' not in data.columns:
        data.rename(columns={'rank_y': 'rank'}, inplace=True)

    st.sidebar.header("Adjust Weights")
    frequency_weight = st.sidebar.slider("Frequency Weight", 0.0, 1.0, 0.4, 0.1)
    page_rank_weight = st.sidebar.slider("Page Rank", 0.0, 1.0, 0.3, 0.1)
    recency_weight = st.sidebar.slider("Recency Weight", 0.0, 1.0, 0.2, 0.1)
    authors_weight = st.sidebar.slider("Authors Weight", 0.0, 1.0, 0.1, 0.1)

    total_weight = frequency_weight + page_rank_weight + recency_weight + authors_weight
    if not math.isclose(total_weight, 1.0, rel_tol=1e-6):
        st.warning("The total weight must sum to exactly 1. Adjust the sliders accordingly.")
        return

    weights = {
        'frequency': frequency_weight,
        'page_rank': page_rank_weight,
        'recency': recency_weight,
        'authors': authors_weight
    }

    ranking_df = get_ranking_df(data, weights, data_selection)
    grid_options = setup_grid_options(ranking_df)

    # Display which data version is being used
    theme_status = "with" if apply_themes else "without"
    st.subheader(f"Top Trending Stories ({theme_status} Themes/Interests)")

    custom_css = """
        <style>
        .ag-theme-streamlit {
            width: 100%; /* Expand table to full width */
            max-width: 1500px; /* Increase the maximum width for the table */
            min-width: 1200px; /* Set a minimum width for better visibility */
            margin: auto; /* Center the table on the page */
            overflow: auto; /* Allow scrolling if necessary */
        }
        </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    AgGrid(
        ranking_df,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        height=600,
        use_container_width=False,
    )

    if st.button('Export to Google Sheets'):
        try:
            url = export_to_gsheet(ranking_df, "Trending Signals Data")  # Pass ranking_df here
            st.success(f'Data exported successfully! [Open Sheet]({url})')
        except Exception as e:
            st.error(f'Error exporting data: {str(e)}')

def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login()
        st.stop()

    show_navigation()

    if st.session_state.get("current_page") == "trending":
        show_trending_scores()
    elif st.session_state.get("current_page") == "aggregator":
        show_aggregator_scores()
    elif st.session_state.get("current_page") == "overall":
        show_overall_view()
    elif st.session_state.get("current_page") == "reddit":
        show_reddit_signals()
    else:
        st.title("Welcome to the Dashboard")
        st.write("Please select an option from the sidebar to begin.")

if __name__ == "__main__":
    print('Starting the application now...')
    main()