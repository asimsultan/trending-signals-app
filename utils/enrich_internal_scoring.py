from collections import Counter
from tqdm import tqdm
import pandas as pd
from get_data_from_es import fetch_data_by_story_id

def get_internal_points(story_id):
    the_keys = {}
    index_name = 'event_map_alias'
    try:
        data = fetch_data_by_story_id(index_name, story_id=story_id)[0]
        for key in data['_source'].keys():
            if key not in ['thumbs', 'sources', 'storyVector', 'description', 'slug', 'themes', 'interests', 'wireStoryRefs', 'place', 'tags', 'title']:
                if key in ['topNews', 'views', 'rank', 'topNewsEligibleEditions', 'mediaCount', 'sourceCount', 'updated']:
                    the_keys[key] = data['_source'][key]
        return the_keys
    except:
        return None

def categorize_rank(rank):
    if rank <= 75:
        return "None"
    elif rank <= 101:
        return "Small Boost"
    elif rank <= 110:
        return "Small-Med Boost"
    elif rank <= 120:
        return "Medium Boost"
    elif rank <= 135:
        return "Large Boost"
    else:
        return "HUGE Boost"

def get_extended_data(df):
    top_news_list = []
    views_list = []
    rank_list = []
    frequency_list = []
    topNewsEligibleEditions_list = []
    mediaCount_list = []
    sourceCount_list = []
    story_frequency = dict(Counter(df['story_id']))
    story_ids = list(df['story_id'].unique())
    dates_list = []

    final_story_ids = []
    for story_id in tqdm(story_ids):
        data = get_internal_points(story_id)
        if data!=None:
            top_news_list.append(data['topNews'])
            views_list.append(data['views'])
            rank_list.append(data['rank'])
            topNewsEligibleEditions_list.append(data['topNewsEligibleEditions'])
            mediaCount_list.append(data['mediaCount'])
            sourceCount_list.append(data['sourceCount'])
            dates_list.append(data['updated'])
            frequency_list.append(story_frequency[story_id])
            final_story_ids.append(story_id)

    print('Total', len(story_ids))
    print('Final', len(final_story_ids))

    test_df = pd.DataFrame(
        {'story_id': final_story_ids, 'date': dates_list, 'frequency': frequency_list, 'top_news': top_news_list, 'views': views_list,
         'rank': rank_list, 'top_designations': topNewsEligibleEditions_list, 'mediaCount': mediaCount_list,
         'sourceCount': sourceCount_list})

    test_df['link'] = test_df['story_id'].apply(lambda x: str('https://ground.news/article/') + x)
    test_df = test_df.sort_values(by='frequency', ascending=False)
    test_df['rank_mask'] = test_df['rank'].apply(categorize_rank)
    test_df = test_df[
        ['story_id', 'date', 'top_news', 'views', 'rank', 'rank_mask', 'top_designations', 'mediaCount', 'sourceCount', 'link']]
    final_df = pd.merge(df, test_df, on='story_id')
    return final_df