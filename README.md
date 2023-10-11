# tape_arxiv_2023

This repository contains scripts for constructing and processing the `arxiv-2023` dataset, which is a sub-repo for [TAPE](https://github.com/XiaoxinHe/TAPE/).




## Background

GPT-3.5’s training data might include certain arXiv papers, given its comprehensive ingestion of textual content from the internet. However, the precise composition of these arXiv papers within GPT-3.5’s training remains undisclosed, rendering it infeasible to definitively identify their inclusion. 

To address this concern, we created a novel dataset `arxiv-2023`. We made sure that this dataset only included papers published in 2023 or later, which is well beyond the knowledge cutoff for GPT-3.5, as it was launched in November 2022. 


We collected all cs arXiv papers published from January 2023 to September 2023 from [arxiv.org](https://arxiv.org/list/cs/recent). We then utilized the [Semantic Scholar  API](https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_get_paper_citations) to retrieve citation relationships. This process yielded a comprehensive graph containing 57,471 papers and 122,835 connections.

## Citation

```
@misc{he2023harnessing,
      title={Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning}, 
      author={Xiaoxin He and Xavier Bresson and Thomas Laurent and Adam Perold and Yann LeCun and Bryan Hooi},
      year={2023},
      eprint={2305.19523},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Data Collection

To collect data from [arXiv](https://arxiv.org/list/cs/recent) and [Semantic Scholar](https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_get_paper_citations), run the following command:
```
python src/collect.py --START 2301 --END 2309
```

This command will:

- Collect cs arXiv papers for the specified range of months (e.g., January 2023 to October 2023).
- Collect citation relationships from Semantic Scholar for the collected papers.
- Save the collected data to the `dataset/arxiv_2023_orig/` directory.


## Data Processing


### Step 1: Download Word2Vec Model

Download the [Word2Vec](https://huggingface.co/fse/word2vec-google-news-300) model from [this link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) and place it in the `MODEL_PATH` directory.


### Step 2: Construct Citation Graph
To process the collected data and construct a citation graph, run the following command:

```
python src/process.py --MODEL_PATH $MODEL_PATH
```

Each node is an arXiv paper and each directed edge indicates that one paper cites another one. Each paper comes with a 300-dimensional feature vector obtained by averaging the embeddings of words in its title and abstract. The embeddings of individual words are computed by running Word2Vec model.

The processed data will be saved as `dataset/arxiv_2023/geometric_data_processed.pt`. 
