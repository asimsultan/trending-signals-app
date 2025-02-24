from elasticsearch import Elasticsearch
import boto3
from botocore.exceptions import ClientError
import json

import warnings
warnings.filterwarnings("ignore")

REGION_NAME = "us-west-2"
session = boto3.Session(region_name=REGION_NAME)

# def get_secret(secret_name):
#     client = session.client(service_name="secretsmanager", region_name=REGION_NAME)
#     try:
#         get_secret_value_response = client.get_secret_value(SecretId=secret_name)
#     except ClientError as e:
#         raise e
#     secret = get_secret_value_response["SecretString"]
#     return json.loads(secret)
#
# prod_secret = get_secret("ES-PROD-Authentication")
# secret = get_secret("ES-Authentication")
# es_vec_domain = secret["es_vec_domain"]
# username = secret["username"]
# password = secret["password"]

es_vec_domain = 'https://es-vec-production.ground.news'
es_vec_username = 'llm_user'
es_vec_password = 'N7aJcxZJ9FKxDLx'


# es_prod_domain = prod_secret["es_domain"]
# prod_user_name = prod_secret["username"]
# prod_password = prod_secret["password"]

# es_domain = 'https://es-production.ground.news'
# es_vec_username = username
# es_vec_password = password

es_prod_domain = 'https://es-production.ground.news'
prod_user_name = 'asim'
prod_password = '81c9&aV%gmewbR5DqbU3'

# es_domain = 'https://es-staging.ground.news'
# es_vec_username = 'asim'
# es_vec_password = 'GwX6pj&S$h1Rn*SxT3hN'


es = Elasticsearch(es_vec_domain, basic_auth=(es_vec_username, es_vec_password))
es_prod = Elasticsearch(
    es_prod_domain,
    basic_auth=(prod_user_name, prod_password),
)

def get_esr_by_url(url, fields=None, sort=None, size=500):
    if fields is None:
        fields = ['id', 'storyId', 'date', 'sourceId', 'url', 'title']

    query = {
        "query": {
            "term": {
                "url": url
            }
        },
        "_source": fields,
        "size": size
    }

    if sort:
        query["sort"] = sort

    result = es.search(
        index="esr_alias",
        body=query
    )

    hits = result["hits"]["hits"]
    if hits:
        hit = hits[0]
        return {
            "id": hit["_id"],
            **hit["_source"]
        }
    return None

def fetch_data_by_story_id(index_name, story_id=None, query=None, size=1):
    if story_id:
        query = {
            "query": {
                "term": {
                    "_id": story_id
                }
            }
        }
    else:
        query = query or {"query": {"match_all": {}}}  # Default to match_all query

    try:
        response = es_prod.search(index=index_name, body=query, size=size)
        hits = response.get("hits", {}).get("hits", [])
        return hits
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []