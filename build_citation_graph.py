import torch
import glob
import json
import pandas as pd
import time


from torch_geometric.data import Data


def main():
    start = time.time()

    files = glob.glob('dataset/paper_info/*.json')
    paperids = []
    paperid2arxivid = {}

    arxiv_ids = []
    titles = []
    abstracts = []
    subjects = []
    for f in files:
        data = json.load(open(f))
        paperid2arxivid[data['paperId']] = data['arxiv_id']
        paperids.append(data['paperId'])
        arxiv_ids.append(data['arxiv_id'])
        titles.append(data['title'])
        abstracts.append(data['abstract'])
        subjects.append(data['subject'])

    df = pd.DataFrame({'arxiv_id': arxiv_ids, 'title': titles,
                      'abstract': abstracts, 'subject': subjects})

    # construct labels
    mapping = pd.read_csv('dataset/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz',
                          compression='gzip', header=0, sep=',', quotechar='"', )
    mapping['arxiv category'] = mapping['arxiv category'].apply(
        lambda x: x.split(' ')[-1])
    category2label = dict(zip(mapping['arxiv category'], mapping['label idx']))
    df['label'] = df['subject'].apply(
        lambda x: category2label[x.split('.')[-1].split(')')[0].lower()])
    df['node_id'] = [i for i in range(len(df))]
    df.to_csv('dataset/arxiv_2023_full.csv', index=False)

    # construct graphs
    print("Constructing a citation graph...")
    arxivid2nodeid = dict(zip(df['arxiv_id'], df['node_id']))
    edges = []
    for f in files:
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
                y=torch.tensor(df['label']), num_nodes=len(df))
    torch.save(data, 'dataset/arxiv_2023.pt')
    print(
        f"Finish constructing a citation graph in {(time.time() - start)/60:.2f} mins")
    print("# nodes: ", data.num_nodes)
    print("# edges: ", data.num_edges)


if __name__ == "__main__":
    main()
