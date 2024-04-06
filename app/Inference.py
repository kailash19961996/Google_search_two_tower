from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import sentencepiece as spm
import gensim
from gensim.models import Word2Vec
import random
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from model.two_tower_classes import TwoTowerNN


vocab_size = 1000
embedding_dim = 100
query_hidden_size = 64
doc_hidden_size = 64
query_num_layers = 2
doc_num_layers = 2
output_size = 1
fine_tune=False

eval_model = TwoTowerNN( 
    sp_model_path='app/m.model', word2vec_model_path='app/word2vec.model',
    embedding_dim=embedding_dim,
    query_hidden_size=query_hidden_size, doc_hidden_size=doc_hidden_size,
    query_num_layers=query_num_layers, doc_num_layers=doc_num_layers,
    output_size=output_size, vocab_size=vocab_size, fine_tune=fine_tune,
)

eval_model.load_state_dict(torch.load('app/two_tower_model.pth'))

# Load document embeddings from the file
with open("app/document_embeddings.json", "r") as f:
    document_embeddings_lists = json.load(f)

# Convert lists to tensors
document_embeds = {i: torch.tensor(embedding) for i, embedding in enumerate(document_embeddings_lists)}


# Load the dictionary from the JSON file
with open("app/id_doc_dict.json", "r") as f:
    id_doc_dict = json.load(f)

def predict_passages(query):
    candidate_docs_ids=id_doc_dict
    k=5
    result = eval_model.predict(query, candidate_docs_ids, id_doc_dict=id_doc_dict, doc_emb_dict=document_embeds, k=k)
    return result


