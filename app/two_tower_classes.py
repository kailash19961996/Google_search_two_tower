
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

# 2-TOWER NN
class TwoTowerNN(nn.Module):
    "TwoTower architecture tries to capture the fact that"
    "queries and documents have different semantic, syntactic structures"
    def __init__(
        self, sp_model_path, word2vec_model_path,
        embedding_dim,#query_input_dim, doc_input_dim,
        query_hidden_size, doc_hidden_size, query_num_layers, doc_num_layers,
        output_size, vocab_size, fine_tune,
    ):
        # setup sentencepiece tokenizer model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(sp_model_path)
        # setup word2vec model
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        # fine_tune parameter
        self.fine_tune = fine_tune
        super(TwoTowerNN, self).__init__()

        # Fine-tune word embeddings
        if self.fine_tune:
          # Embedding layers
          self.query_embedding = nn.Embedding(vocab_size, embedding_dim)
          self.doc_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Towers:
        # Query tower
        self.query_tower = nn.GRU(
            embedding_dim, #if not fine_tune else vocab_size, #embedding_dim, #query_input_size,
            query_hidden_size,
            query_num_layers,
            batch_first=True
        )
        # Document tower
        self.doc_tower = nn.GRU(
            embedding_dim, #if not fine_tune else vocab_size, #embedding_dim, #doc_input_size,
            doc_hidden_size,
            doc_num_layers,
            batch_first=True,
        )

    # forward function
    def forward(self, query_input, doc_input):
        if self.fine_tune:
            # Embedding lookup
            if query_input is not None:
              query_embedded = self.query_embedding(query_input)
            if doc_input is not None:
              doc_embedded = self.doc_embedding(doc_input)
            # RNN forward pass
            if query_embedded is not None:
              _, query_hidden = self.query_tower(query_embedded)
            if doc_embedded is not None:
              _, doc_hidden = self.doc_tower(doc_embedded)
        else:
            # RNN forward pass
            if query_input is not None:
              _, query_hidden = self.query_tower(query_input)
            if doc_input is not None:
              _, doc_hidden = self.doc_tower(doc_input)

        if query_input is None:
          query_embedding = None
        else:
          query_embedding = query_hidden[-1]

        if doc_input is None:
          doc_embedding = None
        else:
          doc_embedding = doc_hidden[-1]

        # Return query and document embeddings
        return query_embedding, doc_embedding


    # Define preprocess_query function
    def prepare(self, text):
        # Tokenize the text using SentencePiece model
        processed_text = self.sp_model.encode_as_ids(text)
        # Vectorize the text using Word2Vec model
        if not self.fine_tune:
          processed_text = [
              self.word2vec_model.wv.get_vector(tok)for tok in processed_text
          ]
        return processed_text


    # get document embeddings
    def get_document_embeddings(self, documents):
        # Preprocess candidate documents
        processed_documents = [
            self.prepare(doc) for doc in documents
        ]

        # Encode candidate documents
        document_embeddings = []
        for doc_input in processed_documents:
            doc_input = torch.tensor(doc_input)  # Convert to tensor if required
            _, doc_embedding = self.forward(None, doc_input)
            document_embeddings.append(doc_embedding)

        return document_embeddings


    # predict function
    def predict(
        self, new_query, candidate_docs_ids, doc_emb_dict, id_doc_dict, k,
    ):
        # Preprocess the new query
        query_input = self.prepare(new_query)

        # Encode the query
        query_input = torch.tensor(query_input)  # Convert to tensor if required
        query_embedding, _ = self.forward(query_input, None)

        # get document encodings/embeddings
        candidate_documents = [
            id_doc_dict[doc_id] for doc_id in candidate_docs_ids
        ]
        docs_in_dict = list(
            set(candidate_documents).intersection(set(doc_emb_dict.keys()))
        )
        docs_not_in_dict = list(
            set(candidate_documents) - set(doc_emb_dict.keys())
        )

        document_embeddings = {}
        if len(docs_in_dict) > 0:
          document_embeddings.update(
              {doc: doc_emb_dict[doc] for doc in docs_in_dict}
          )
        if len(docs_not_in_dict) > 0:
          can_doc_embeddings = self.get_document_embeddings(
              docs_not_in_dict
          )
          document_embeddings.update(
              {
                  doc: emb for doc, emb in zip(
                      docs_not_in_dict, can_doc_embeddings,
                  )
              }
          )

        # Calculate cosine similarity scores
        similarity_scores = [
          F.cosine_similarity(
              query_embedding.unsqueeze(0), doc_embedding.unsqueeze(0), dim=1
          ).item() for doc_embedding in document_embeddings.values()
        ]

        # Rank documents by similarity
        ranked_documents = sorted(zip(document_embeddings.keys(), similarity_scores),key=lambda x: x[1], reverse=True)

        # Retrieve top-K documents
        top_k_documents = ranked_documents[:k]

        return top_k_documents

class TwoTowerDataset(Dataset):
    def __init__(
        self, triples, id_doc_dict,
        tokenized_id_sentences, token_id_embeddings, fine_tune,
    ):
        self.triples = triples
        self.doc_dict = id_doc_dict
        self.tokenized_id_sentences = tokenized_id_sentences
        self.token_id_embeddings = token_id_embeddings
        self.fine_tune = fine_tune

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        query, positive_indx, negative_indx = self.triples[idx]
        positive_doc = self.doc_dict[positive_indx]
        negative_doc = self.doc_dict[negative_indx]
        # if not fine_tune take the collection of token embeddings
        if not self.fine_tune:
          # query
          tok_query = self.tokenized_id_sentences[query]
          query = np.array(
              [self.token_id_embeddings[tok] for tok in tok_query]
          )
          # postive doc
          tok_pos_doc = self.tokenized_id_sentences[positive_doc]
          positive_doc = np.array(
              [self.token_id_embeddings[tok] for tok in tok_pos_doc]
          )
          # negative doc
          tok_neg_doc = self.tokenized_id_sentences[negative_doc]
          negative_doc = np.array(
              [self.token_id_embeddings[tok] for tok in tok_neg_doc]
          )

        # if fine_tune take the tokenized sentences
        else:
          query = self.tokenized_id_sentences[query]
          positive_doc = self.tokenized_id_sentences[positive_doc]
          negative_doc = self.tokenized_id_sentences[negative_doc]

        return (
            query,
            positive_doc,
            negative_doc,
        )

def collate_fn(batch, fine_tune):
    # MAKE ALL SAME SHAPE!!
    # if training token embeddings within we need int else float
    if fine_tune:
      torch_type=torch.int64
    else:
      torch_type=torch.float32
    # Sort batch by sequence length (optional but can improve efficiency)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    # Extract context sequences and target indices
    queries, pos_docs, neg_docs = zip(*batch)

    # Pad context sequences to the length of the longest sequence in the batch
    padded_queries = pad_sequence(
        [torch.tensor(query, dtype=torch_type) for query in queries],
        batch_first=True, padding_value=0,
      )
    padded_pos_docs = pad_sequence(
        [torch.tensor(pos_doc, dtype=torch_type) for pos_doc in pos_docs],
        batch_first=True, padding_value=0,
    )
    padded_neg_docs = pad_sequence(
        [torch.tensor(neg_doc, dtype=torch_type) for neg_doc in neg_docs],
        batch_first=True, padding_value=0,
    )

    return padded_queries, padded_pos_docs, padded_neg_docs
