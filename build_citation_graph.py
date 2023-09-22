from torch_geometric.data import Data
import torch
import glob
import json
import pandas as pd


def main():
    files = glob.glob('data/paper_info/*.json')
    paperids = []
    paperid2arxivid = {}

    arxiv_ids = []
    titles = []
    abstracts = []
    categories = []
    for f in files:
        data = json.load(open(f))
        paperid2arxivid[data['paperId']] = data['arxiv_id']
        paperids.append(data['paperId'])
        arxiv_ids.append(data['arxiv_id'])
        titles.append(data['title'])
        abstracts.append(data['abstract'])
        categories.append(data['category'])

    df = pd.DataFrame({'arxiv_id': arxiv_ids, 'title': titles,
                       'abstract': abstracts, 'category': categories})

    # construct labels
    mapping = pd.read_csv('dataset/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz',
                          compression='gzip', header=0, sep=',', quotechar='"', )
    mapping['arxiv category'] = mapping['arxiv category'].apply(
        lambda x: x.split(' ')[-1])
    category2label = dict(zip(mapping['arxiv category'], mapping['label idx']))
    df['label'] = df['category'].apply(
        lambda x: category2label[x.split('.')[-1].split(')')[0].lower()])
    df['node_id'] = [i for i in range(len(df))]
    df.to_csv('data/paper_info.csv', index=False)

    # construct graphs
    arxivid2nodeid = dict(zip(df['arxiv_id'], df['node_id']))
    edges = []
    for f in (files):
        data = json.load(open(f))

        for r in data['references']:
            if r['paperId'] in paperid2arxivid:
                src = arxivid2nodeid[data['arxiv_id']]
                dst = arxivid2nodeid[paperid2arxivid[r['paperId']]]
                edges.append((src, dst))

        for c in data['citations']:
            if c['paperId'] in paperid2arxivid:
                src = arxivid2nodeid[paperid2arxivid[c['paperId']]]
                dst = arxivid2nodeid[data['arxiv_id']]
                edges.append((src, dst))

    data = Data(edge_index=torch.tensor(edges).t(),
                y=torch.tensor(df['label']))
    torch.save(data, 'data/arxiv_2023.pt')
    print("# nodes: ", data.num_nodes)
    print("# edges: ", data.num_edges)


if __name__ == "__main__":
    main()
