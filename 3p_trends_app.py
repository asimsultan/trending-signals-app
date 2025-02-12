import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
            # Initialize page state after successful login
            if "current_page" not in st.session_state:
                st.session_state["current_page"] = None
        else:
            st.session_state["authenticated"] = False
            st.sidebar.error("Invalid credentials")


def show_navigation():
    st.sidebar.header("Navigation")
    if st.sidebar.button("Trending Scores"):
        st.session_state["current_page"] = "trending"
        st.rerun()
    if st.sidebar.button("Aggregator Scores"):
        st.session_state["current_page"] = "aggregator"
        st.rerun()


# Your existing functions remain the same
def preprocess_dates(date_col):
    return date_col.apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) and hasattr(x, 'tzinfo') else x)


def calculate_recency(new_date, current_date):
    if pd.isnull(new_date):
        return 0
    days_diff = (current_date - new_date).days
    recency_score = 1 / (1 + np.exp(days_diff / 30))
    return recency_score


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
        date=('new_date', 'first'),
        num_authors=('authors', lambda x: x.apply(len).sum()),
        page_rank=('page_rank', 'mean'),
        views=('views', 'first'),
        internal_rank=('rank', 'first'),
        rank_mask=('rank_mask', 'first'),
        top_designations=('top_designations', 'first'),
        mediaCount=('mediaCount', 'first'),
        sourceCount=('sourceCount', 'first')

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
        ['story_id', 'title', 'frequency', 'trending_signal_score', 'page_rank_norm', 'views', 'internal_rank', 'rank_mask','top_designations', 'mediaCount', 'sourceCount', 'date']]
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


def get_latest_available_data(selection):
    max_days_to_check = 7  # Number of past days to check if today's data isn't available
    bucket_name = "trending-signal-bucket"  # Correct bucket name without the date
    for i in range(max_days_to_check):
        check_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        object_key = f"{check_date}/Business_df.pkl" if selection == "Business" else f"{check_date}/Australia_df.pkl"
        print(f"Checking: {bucket_name}/{object_key}")
        try:
            data = read_pkl_from_s3(bucket_name, object_key)
            if data is not None:
                return data
        except Exception as e:
            print(f"Error for {object_key}: {e}")
            continue

    print("No data available for the past 7 days.")
    return None

@st.cache_data
def load_data(selection):
    return get_latest_available_data(selection)


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
        maxWidth=280,
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
        "domLayout": "autoHeight",  # Flexible height
        "animateRows": True,  # Enables row animation
        "suppressRowVirtualisation": True,  # Renders all rows
        "rowBuffer": 10,  # Buffer for smooth scrolling
        "cellFlashDuration": 700,  # Flash duration for cell changes
        "cellFadeDuration": 1000,  # Fade-out duration for flash
        "ensureDomOrder": True,  # Accessibility feature
        "suppressMaxRenderedRowRestriction": True,  # Render more than 500 rows
        "suppressAutoSize": True,  # Prevent columns from resizing automatically
        "defaultColDef": {"suppressSizeToFit": True},  # Prevent individual column resizing
    })
    return grid_options


def show_trending_scores():
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
        return

    weights = {
        'frequency': frequency_weight,
        'page_rank': page_rank_weight,
        'recency': recency_weight,
        'authors': authors_weight
    }

    ranking_df = get_ranking_df(data, weights, data_selection)
    grid_options = setup_grid_options(ranking_df)

    st.subheader("Top Trending Stories")

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


def show_aggregator_scores():
    st.title("Aggregator Scores")

    bucket_name = "trending-signal-bucket"
    object_name = '2025-02-11/Aggregated_results_feb12.pkl'
    data = read_pkl_from_s3(bucket_name, object_name)
    # Create grid options builder
    gb = GridOptionsBuilder.from_dataframe(data)

    # Configure columns
    gb.configure_column("story_id", header_name="Story ID", width=150)
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
        data,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        height=600,
        use_container_width=False
    )

def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login()
        st.stop()

    # Show navigation buttons after successful login
    show_navigation()

    # Display the selected page
    if st.session_state.get("current_page") == "trending":
        show_trending_scores()
    elif st.session_state.get("current_page") == "aggregator":
        show_aggregator_scores()
    else:
        st.title("Welcome to the Dashboard")
        st.write("Please select Trending Scores or Aggregator Scores from the sidebar to begin.")

if __name__ == "__main__":
    main()