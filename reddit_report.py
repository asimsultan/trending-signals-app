import pandas as pd
import boto3, io, json, joblib, pickle
from fpdf import FPDF
from get_ranking_for_report_evaluation import get_ranking_df

BUCKET_NAME = "trending-signal-bucket"

def read_the_data(bucket_name: str, filename: str, current_date, _type: str):
    try:
    # current_date = "2025-03-17"
        object_key = f"{current_date}/{filename}"

        print('Trying to read', object_key)
        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        content = obj['Body'].read()
        bytestream = io.BytesIO(content)
        if _type == 'json':
            content = content.decode('utf-8')
            data = json.loads(content)
        elif _type == 'joblib':
            data = joblib.load(bytestream)
        elif _type == 'pickle':
            data = pickle.load(bytestream)
        return data
    except Exception as e:
        print(f"Read error: {e}")
        return None

def get_latest_available_data(file_name_header, _type):
    max_days_to_check = 30
    for i in range(max_days_to_check):
        check_date = (pd.Timestamp.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            data_dict = read_the_data(BUCKET_NAME, file_name_header, check_date, _type)
            if data_dict is not None:
                return data_dict
        except Exception as e:
            print(f"Error for {file_name_header}: {e}")
            continue

from datetime import datetime, timedelta

def filter_by_date_time(df):
    df['date'] = pd.to_datetime(df['date'])
    now_utc = datetime.utcnow()
    cutoff_time = now_utc - timedelta(hours=96)
    cutoff_time = pd.to_datetime(cutoff_time).tz_localize(None)
    df['date'] = df['date'].dt.tz_localize(None)
    df = df[df['date'] >= cutoff_time]
    return df

def get_ranked_df(weights, df, category):
    ranked_df = get_ranking_df(df, weights, category)
    ranked_df = pd.merge(ranked_df, df, on=['story_id', 'title'])
    ranked_df = ranked_df[['story_id', 'title', 'trending_signal_score', 'rank_mask', 'link', 'date']]
    ranked_df = filter_by_date_time(ranked_df)
    return ranked_df

def get_non_boosted_stories(df, story_col, date_col):
    df = df[df.index<50][[story_col , 'title', 'rank_mask', 'link', date_col]]
    df = df[df['rank_mask'] == 'None'][[story_col, 'title', 'link', date_col]]
    df.drop_duplicates(subset=story_col, inplace=True)
    return df


def clean_for_latin1(text):
    """Replace all problematic Unicode characters for FPDF compatibility"""
    replacements = {
        '\u201c': '"',  # opening smart quote
        '\u201d': '"',  # closing smart quote
        '\u2018': "'",  # opening smart single quote
        '\u2019': "'",  # closing smart single quote (apostrophe)
        '\u2013': '-',  # en dash
        '\u2014': '--',  # em dash
        '\u2026': '...',  # ellipsis
        '\u20ac': 'EUR',  # euro symbol
        # Add more replacements as needed
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # As a last resort, encode and decode as ASCII with error handling
    return text.encode('ascii', 'replace').decode('ascii')

def generate_combined_report(weights, reddit_file_path, business_file_path, australia_file_path, aggregators_file_path, output_path="combined_report.pdf"):
    reddit_df = get_latest_available_data(reddit_file_path, "pickle")
    trending_business_df = get_latest_available_data(business_file_path, "pickle")
    trending_australian_df = get_latest_available_data(australia_file_path, "pickle")
    trending_aggregator_df = get_latest_available_data(aggregators_file_path, "pickle")

    numeric_cols = ['compositeScore', 'upvotes', 'comments', 'engagementScore', 'rank']
    for col in numeric_cols:
        reddit_df[col] = pd.to_numeric(reddit_df[col], errors='coerce')
    reddit_df.sort_values(by='compositeScore', ascending=False, inplace=True)
    reddit_df.reset_index(drop=True, inplace=True)

    australian_ranked_df = get_ranked_df(weights, trending_australian_df, 'Australia')
    business_ranked_df = get_ranked_df(weights, trending_business_df, 'Business')

    reddit_stories_to_be_boosted = get_non_boosted_stories(reddit_df, story_col='storyId', date_col='createdAt')
    trending_business_stories_to_be_boosted = get_non_boosted_stories(business_ranked_df, story_col='story_id',
                                                                      date_col='date')
    trending_australian_stories_to_be_boosted = get_non_boosted_stories(australian_ranked_df, story_col='story_id',
                                                                        date_col='date')

    # reddit_filtered_df = reddit_df[reddit_df.index < 50][['storyId', 'title', 'rank_mask', 'link']]
    # reddit_filtered_df = reddit_filtered_df[reddit_filtered_df['rank_mask'] == 'None'][['storyId', 'title', 'link']]

    def categorize_source(count):
        if count == 1:
            return "None-1Source"
        elif 2 <= count <= 5:
            return "None-2to5Sources"
        elif 6 <= count <= 10:
            return "None-5to10Sources"
        else:
            return "None-moreThan10Sources"

    high_boost_low_composite = reddit_df[(reddit_df['rank_mask'] == 'HUGE Boost') & (reddit_df.index > 49)]
    none_top_50 = reddit_df[(reddit_df['rank_mask'] == 'None') & (reddit_df.index <= 49)]
    none_source_count = none_top_50['sourceCount_y'].value_counts().sort_index()

    the_df = reddit_df[(reddit_df['rank_mask'] == 'None') & (reddit_df.index < 49)]
    the_df["sourceCount_group"] = the_df["sourceCount_x"].apply(categorize_source)
    none_source_group_counts = the_df["sourceCount_group"].value_counts()

    # --- Business Data Analysis ---
    # trending_business_df = read_the_data("trending-signal-bucket", business_file_path, "pickle")
    weights = {'frequency': 0.5, 'recency': 0.4, 'authors': 0.1}
    ranked_business_df = get_ranking_df(trending_business_df, weights, 'Business')
    ranked_business_df = pd.merge(ranked_business_df, trending_business_df, on='story_id')
    ranked_business_df = ranked_business_df.rename(columns={'title_x': 'title'})
    df_sorted = ranked_business_df.copy()
    top_50 = df_sorted.head(50)

    top_50_boost_counts = top_50['rank_mask'].value_counts()
    all_boost_counts = ranked_business_df['rank_mask'].value_counts()

    false_positives = top_50[(top_50['rank_mask'] == 'None') | (top_50['rank_mask'] == 'Small Boost') | (top_50['rank_mask'] == 'Small-Med Boost')]
    false_negatives = ranked_business_df[(ranked_business_df['rank_mask'].isin(['Large Boost', 'HUGE Boost'])) & (ranked_business_df.index >= 50)]

    top_50_none_count = len(top_50[top_50['rank_mask'] == 'None'])
    huge_boost_not_top_50_count = len(ranked_business_df[(ranked_business_df['rank_mask'] == 'HUGE Boost') & (ranked_business_df.index >= 50)])

    # --- PDF Report Generation ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=24)
    pdf.cell(0, 50, "Trending Signals Report", ln=True, align='C')
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Generated on: " + pd.Timestamp.now().strftime('%Y-%m-%d'), ln=True, align='C')
    pdf.ln(30)

    # Reddit Section
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(0, 10, "Reddit Trending Signals Analysis", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Stories that are not boosted and should be boosted - From Reddit Signals", ln=True)
    pdf.ln(5)

    if not reddit_stories_to_be_boosted.empty:
        pdf.set_font("Arial", size=8)  # Reduced font size
        pdf.cell(55, 8, "Story ID", border=1)  # Reduced Story ID width
        pdf.cell(90, 8, "Title", border=1)  # Reduced Title width
        pdf.cell(35, 8, "Link", border=1, ln=True)  # Increased Link width

        for index, row in reddit_stories_to_be_boosted.iterrows():
            # title = row['title'].replace('\u2019', "'").replace('\u2013', '-').replace('\u201c','"').replace('\u201d','"') #replace all problematic characters.
            # title = row['title'].replace('\u20ac', 'EUR')
            # title = row['title'].replace('\u2019', "'")
            title = clean_for_latin1(row['title'])
            pdf.cell(55, 8, row['storyId'], border=1)
            pdf.cell(90, 8, title[:65], border=1)  # Limit title length
            pdf.cell(35, 8, row['link'][:28], border=1, ln=True)  # Limit link length

    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "No stories found that need boosting.", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Stories that are not boosted and should be boosted - From Business VIP Signals", ln=True)
    pdf.ln(5)

    if not trending_business_stories_to_be_boosted.empty:
        pdf.set_font("Arial", size=8)  # Reduced font size
        pdf.cell(55, 8, "Story ID", border=1)  # Reduced Story ID width
        pdf.cell(90, 8, "Title", border=1)  # Reduced Title width
        pdf.cell(35, 8, "Link", border=1, ln=True)  # Increased Link width

        for index, row in trending_business_stories_to_be_boosted.iterrows():
            # title = row['title'].replace('\u2019', "'").replace('\u2013', '-').replace('\u201c','"').replace('\u201d','"') #replace all problematic characters. #replace both problematic characters.
            # title = row['title'].replace('\u20ac', 'EUR')
            # title = row['title'].replace('\u2019', "'")
            title = clean_for_latin1(row['title'])
            pdf.cell(55, 8, row['story_id'], border=1)
            pdf.cell(90, 8, title[:65], border=1)  # Limit title length
            pdf.cell(35, 8, row['link'][:28], border=1, ln=True)  # Limit link length

    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "No stories found that need boosting.", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Stories that are not boosted and should be boosted - From Australian VIP Signals", ln=True)
    pdf.ln(5)

    if not trending_australian_stories_to_be_boosted.empty:
        pdf.set_font("Arial", size=8)  # Reduced font size
        pdf.cell(55, 8, "Story ID", border=1)  # Reduced Story ID width
        pdf.cell(90, 8, "Title", border=1)  # Reduced Title width
        pdf.cell(35, 8, "Link", border=1, ln=True)  # Increased Link width

        for index, row in trending_australian_stories_to_be_boosted.iterrows():
            # title = row['title'].replace('\u2019', "'").replace('\u2013', '-').replace('\u201c','"').replace('\u201d','"') #replace all problematic characters. #replace both problematic characters.
            # title = row['title'].replace('\u20ac', 'EUR')
            # title = row['title'].replace('\u2019', "'")
            title = clean_for_latin1(row['title'])
            pdf.cell(55, 8, row['story_id'], border=1)
            pdf.cell(90, 8, title[:65], border=1)  # Limit title length
            pdf.cell(35, 8, row['link'][:28], border=1, ln=True)  # Limit link length

    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "No stories found that need boosting.", ln=True)

    pdf.ln(10)


    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Stories with High Boost and low composite score (not in top 50): {len(high_boost_low_composite)}", ln=True)
    pdf.cell(0, 10, f"Stories with None rank_mask and in the top 50: {len(none_top_50)}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Top 50 'None' Stories by Source Count", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    for index, value in none_source_count.items():
        pdf.cell(40, 8, f"Source Count {index}:", border=1)
        pdf.cell(30, 8, str(value), border=1, ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Source Count Distribution for 'HUGE Boost' Stories (Not Top 50)", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    for index, value in none_source_group_counts.items():
        pdf.cell(60, 8, f"{index}:", border=1)
        pdf.cell(30, 8, str(value), border=1, ln=True)

    # Business Section
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(0, 10, "Business Trending Signals Analysis", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Overall Boost Distribution:", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    for boost, count in all_boost_counts.items():
        pdf.cell(0, 8, f"{boost}: {count}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Top 50 Trending Signals Analysis:", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "This table shows the distribution of boost levels for the top 50 trending signals, highlighting the discrepancies between system ranking and admin boosts.", ln=True)
    pdf.cell(0, 8, "It helps identify stories that were highly ranked by our algorithm but overlooked or under-boosted by human admins.", ln=True)
    pdf.ln(5)

    pdf.cell(60, 8, "Boost Level", border=1)
    pdf.cell(30, 8, "Count", border=1, ln=True)
    for boost, count in top_50_boost_counts.items():
        pdf.cell(60, 8, boost, border=1)
        pdf.cell(30, 8, str(count), border=1, ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "False Positives (Top 50 with Low/No Boost):", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Stories ranked high by our system (in top 50) but ignored or under-boosted by human admins.",
             ln=True)
    pdf.cell(0, 8, f"Number of False Positives: {len(false_positives)}", ln=True)

    pdf.ln(10)

    # False Negatives
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "False Negatives (Missed High Boost Stories):", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Important trending stories missed by our system (large/huge boost but not in top 50).", ln=True)
    pdf.cell(0, 8, f"Number of False Negatives: {len(false_negatives)}", ln=True)

    pdf.ln(10)

    # Additional Insights
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Additional Insights:", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Stories in top 50 with 'None' boost: {top_50_none_count}", ln=True)
    pdf.cell(0, 8, f"Stories with 'HUGE Boost' not in top 50: {huge_boost_not_top_50_count}", ln=True)

    # Save PDF
    pdf.output("business_trending_report.pdf")
    print("Report generated: business_trending_report.pdf")


def generate_report():
    reddit_file_path = "Reddit_signals.pkl"
    business_file_path = "Business_df.pkl"
    australia_file_path = "Australia_df.pkl"
    aggregators_file_path = "Aggregated_results.pkl"

    weights = {}

    weights['frequency'] = 0.5
    weights['recency'] = 0.4
    weights['authors'] = 0.1
    print('Generating Report')
    generate_combined_report(weights, reddit_file_path, business_file_path, australia_file_path, aggregators_file_path, output_path="combined_report.pdf")
    print('Report Generated')