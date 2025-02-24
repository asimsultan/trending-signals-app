from urllib.parse import urlparse, urlunparse
from tqdm import tqdm
from get_data_from_es import get_esr_by_url

def remove_query_params(urls_list):
    clean_list = []
    for url in tqdm(urls_list):
        parsed_url = urlparse(url)
        url_without_query = urlunparse(parsed_url._replace(query=""))
        clean_list.append(url_without_query)
    return clean_list

#this function takes url and returns the story_id
def get_data_dict(urls_list):
    data_dict = {}
    for url in tqdm(urls_list):
        response = get_esr_by_url(url)
        if response:
            data_dict[url] = response
        else:
            data_dict[url] = 'No Articles Found'
    return data_dict