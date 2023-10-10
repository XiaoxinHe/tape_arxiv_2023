import torch
import glob
import json
import pandas as pd
import numpy as np
import time
import gensim
import argparse

from torch_geometric.data.data import Data


def load_word2vec_model(MODEL_PATH):

    model = gensim.models.KeyedVectors.load_word2vec_format(
        MODEL_PATH, binary=True)
    return model


def word2vec(model, text, hidden_dim=300):
    words = text.split()  # Tokenize the text into words
    word_vectors = []

    for word in words:
        try:
            vector = model[word]  # Get the Word2Vec vector for the word
            word_vectors.append(vector)
        except KeyError:
            # Handle the case where the word is not in the vocabulary
            pass

    if word_vectors:
        # Calculate the mean of word vectors to represent the text
        text_vector = sum(word_vectors) / len(word_vectors)
    else:
        # Handle the case where no word vectors were found
        text_vector = np.zeros(hidden_dim)

    return text_vector


def main(args):
    start = time.time()
    files = glob.glob(f'dataset/arxiv_2023_orig/paper_info/*.json')
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
    df['node_id'] = [i for i in range(len(df))]

    print("Constructing a citation graph...")

    # construct nodes
    model = load_word2vec_model(args.MODEL_PATH)
    x = [word2vec(model, f"Title: {ti}\n Abstract: {ab}")
         for ti, ab in zip(titles, abstracts)]
    x = torch.Tensor(np.array(x))

    # construct edges
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
    edge_index = torch.tensor(edges).t()

    # construct labels
    mapping = pd.read_csv('dataset/arxiv_2023/mapping/labelidx2arxivcategeory.csv.gz',
                          compression='gzip', header=0, sep=',', quotechar='"', )
    mapping['arxiv category'] = mapping['arxiv category'].apply(
        lambda x: x.split(' ')[-1])
    category2label = dict(zip(mapping['arxiv category'], mapping['label idx']))
    df['label'] = df['subject'].apply(
        lambda x: category2label[x.split('.')[-1].split(')')[0].lower()])
    y = torch.tensor(df['label'])

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(df))
    torch.save(data, f'dataset/arxiv_2023/geometric_data_processed.pt')
    df.to_csv(f'dataset/arxiv_2023_orig/paper_info.csv', index=False)

    print(
        f"Finish constructing a citation graph in {(time.time() - start)/60:.2f} mins")
    print("# nodes: ", data.num_nodes)
    print("# edges: ", data.num_edges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_PATH',
                        type=str,
                        default="~/word2vec/GoogleNews-vectors-negative300.bin.gz")
    args = parser.parse_args()
    main(args)
