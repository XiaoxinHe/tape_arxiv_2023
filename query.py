import json
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import concurrent.futures


def craw_single_page(base_url, skip=0, show=25):
    # Define the URL of the webpage you want to scrape
    url = f"{base_url}?skip={skip}&show={show}"

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    arxiv_ids = []
    titles = []
    subjects = []

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("Successfully retrieved the webpage: ", url)
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all items in the source code
        items_dd = soup.find_all('dd')
        items_dt = soup.find_all('dt')
        assert len(items_dd) == len(items_dt)
        # Loop through the items and extract information
        for dd, dt in zip(items_dd, items_dt):
            # Extract the title
            title_element = dd.find('div', class_='list-title mathjax')
            title = title_element.text.strip().replace("Title:", "").strip()
            titles.append(title)

            # Find the primary subject element and extract the primary subject text
            primary_subject_element = dd.find('span', class_='primary-subject')
            primary_subject = primary_subject_element.text.strip()
            subjects.append(primary_subject)

            arxiv_id_element = dt.find('span', class_='list-identifier')
            arxiv_id = arxiv_id_element.a['href'].split('/')[-1]
            arxiv_ids.append(arxiv_id)
        df = pd.DataFrame(
            {'arxiv_id': arxiv_ids, 'title': titles, 'subject': subjects})
        date = base_url.split('/')[-1]
        df.to_csv(f'arxiv{date}_skip{skip}.csv', index=False, header=False)

    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)


def find_total_entries(base_url):
    # Send an HTTP GET request to the URL
    response = requests.get(base_url)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find the total number of entries
    total_entries = int(soup.find('small').text.split('of')
                        [-1].split('entries')[0].strip())
    return total_entries


def get_arxiv_2023():
    # Define the base URL
    # for i in range(2301, 2309):
    for i in range(2309, 2310):
        base_url = f'https://arxiv.org/list/cs/{i}'

        # Define the base URL and starting skip value
        skip = 0
        items_per_page = 1000  # Number of items per page
        total_entries = find_total_entries(base_url)

        # Loop through the pages and collect paper information
        for _ in tqdm(range(0, total_entries, items_per_page)):
            craw_single_page(base_url=base_url, skip=skip, show=items_per_page)
            skip += items_per_page


def get_citations():

    df = pd.read_csv('data/all_cs.csv',
                     names=['arxiv_id', 'title', 'category'])

    for ind in tqdm(df.index):
        arxiv_id = df['arxiv_id'][ind]

        path = f'data/paper_info/{arxiv_id}.json'
        isExist = os.path.exists(path)
        if isExist:
            return
        url = f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=title,abstract,citations'
        response = requests.get(url)
        if response.status_code == 200:
            response = response.json()
            response['arxiv_id'] = arxiv_id
            response['category'] = df['category'][ind]
            json.dump(response, open(path, 'w'), indent=4)
        else:
            print('Fail to get paper info: ', url)


def query_api(arxiv_id, category):
    path = f'data/paper_info/{arxiv_id}.json'
    isExist = os.path.exists(path)
    if isExist:
        return
    url = f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=title,abstract,citations,references'
    S2_API_KEY = "TYZHUNFiQj7TGXBSdJerG1PHejGg3ffC6fZ2jRY6"
    response = requests.get(url, headers={'X-API-KEY': S2_API_KEY})
    if response.status_code == 200:
        response = response.json()
        response['arxiv_id'] = arxiv_id
        response['category'] = category
        json.dump(response, open(path, 'w'), indent=4)
    else:
        print('Fail to get paper info: ', url)


if __name__ == "__main__":
    # get_arxiv_2023()
    df = pd.read_csv('data/all_cs.csv',
                     names=['arxiv_id', 'title', 'category'])
    arxiv_ids = df['arxiv_id'].tolist()
    categories = df['category'].tolist()

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        res = executor.map(query_api, arxiv_ids, categories)
