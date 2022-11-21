import os
import sys

sys.path.insert(0, os.getcwd())

import pickle
import torch
import csv

from Code.blink.indexer.faiss_indexer import DenseHNSWFlatIndexer
import numpy as np
import torch
from Code.FileIOManager import ModelFileIO, time_format, str_to_tensor
from Code.blink.Load_fine_tuned_model import load_model
from Code.blink.Dataset_processing import str_to_tensor
from Code.blink.biencoder import BiEncoderRanker
from Code.blink.Dataset_processing import filter_entities_for_description, write_complete_entity_dataset, \
    read_complete_entity_dataset
from Code.Logger import log
import requests
import faiss
import json
import re
import time


def instantiate_model(fman):
    log("Reading the arguments...", 0)
    model_name = "biencoder_wiki_large.bin"
    kge_emb_dim = 100

    log("Reading config file of the pretrained model", 0)
    import json
    config_name = "/biencoder_wiki_large.json"
    models_path = fman.models_repo

    with open(f"{models_path}{config_name}") as j_file:
        params = json.load(j_file)
        params["kge_emb_dim"] = kge_emb_dim
    j_file.close()


    reranker = BiEncoderRanker(params)
    log("Loading the model's weights...", 0)
    reranker.load_model(f"{models_path}/BLINK_output/epoch_6/pytorch_model.bin")

    return reranker

def compute_new_embeddings(fman, model, data, t=None):
    from Code.blink.data_process import get_candidate_representation

#    file = open(f"{fman.results_repo}/trained_blink_emb_20_11_22.tsv", "w", encoding="utf-8")
#    writer = csv.writer(file, delimiter="\t")

#    url_map_file = open(f"{fman.results_repo}/final_trained_blink_url_map_20_11_22.pkl", "wb")
    urls_file_content = []

    new_embeddings = []

    token_ids_batch = None
    url_batch = []
    kge_batch = None

    bp = 0

    start = time.time()
    for url in data:

        abstract = data[url][1]
        label = data[url][0]
        emb = torch.FloatTensor(data[url][2])

        urls_file_content.append(url)
        url_batch.append(url)

        desc_token_ids = get_candidate_representation(abstract, model.tokenizer, 196, label)["ids"]
        desc_token_ids = torch.IntTensor(desc_token_ids).unsqueeze(0)

        if token_ids_batch is not None:
            token_ids_batch = torch.cat((token_ids_batch, desc_token_ids), dim=0)
        else:
            token_ids_batch = desc_token_ids

        if kge_batch is not None:
            kge_batch = torch.cat((kge_batch, emb.unsqueeze(0)), dim=0)
        else:
            kge_batch = emb.unsqueeze(0)

        if token_ids_batch.size(0) == 16:
            out = model.encode_candidate(token_ids_batch.to(model.device), kge_emb=kge_batch.to(model.device))

            out.cpu().detach()
            token_ids_batch.cpu().detach()


            if bp % 5000 == 0:
                log(f"Encoded the candidates to a vector of size ({out.size()}) ==> out[idx] = {out[1].size()}", 1)
                log(f"iterating over the url batch taht contains {len(url_batch)} links", 1)

            for u in range(len(url_batch)):
                new_embeddings.append(out[u].tolist())
#                writer.writerow([url_batch[u]]+out[u].tolist())
#                if bp ==1:
#                  log(f"Writing to file {url_batch[u]} , {len(out[u].tolist())} //// len(row)={len([url_batch[u]]+out[u].tolist())}", 3)

            if bp % 5000 == 0:
                log(f"new embeddings got new entries ===> batchs processed = ({bp})", 1)

            end = time.time()
            if bp % 5000 == 0:
                log(f"from {t}: ids_batch = {token_ids_batch.size()} --- time/batch = {time_format(int(end - start))}", 1)
                print("__________________________________________________________________________________________________")

            kge_batch = None
            token_ids_batch = None
            url_batch = []
            bp += 1

#    pickle.dump(urls_file_content, url_map_file)
    print(f"returning from {t} with {len(new_embeddings)}")
    return new_embeddings


def write_index_file(fman, embeddings=None):
#    file = open(f"{fman.results_repo}/trained_blink_emb_20_11_22.tsv", "r", encoding="utf-8")
#    reader = csv.reader(file, delimiter="\t")

#    log("reading embeddings from file...", 1)

#    embeddings = []
#    for row in reader:
#        if len(row)>2:
#          embeddings.append([float(e) for e in row[1:]])
#        if len(embeddings) % 50000 == 0:
#          log(f"processed {len(embeddings)} lines", 2)

    embeddings = np.array(embeddings, dtype=np.float32)

    log(f"indexing data ...", 1)
    index = DenseHNSWFlatIndexer(1024, store_n=64, ef_search=32, ef_construction=32)
    index.index_data(embeddings)

    log(f"Writing index to file => {fman.results_repo}/final_trained_blink_emb_21_11")
    index.serialize(f"{fman.results_repo}/final_trained_blink_emb_21_11")


if __name__ == "__main__":

    log("Instantiating the FileIO instance", 0)
    file_manager = ModelFileIO()

    if not os.path.exists(f"{file_manager.results_repo}/final_trained_blink_emb_21_11_22.tsv"):

      model = instantiate_model(file_manager)

      log(f"Reading the complete dataset of entity desc...")
      f = open("./Data/Results/Complete_17_11_22.tsv", "r", encoding="utf-8")
      reader = csv.reader(f, delimiter="\t")

      ent_full_data = {}
      c = 0
      for row in reader:
          if len(row)>2:
              c += 1
              ent_full_data[row[0]] = [row[2], row[3], str_to_tensor(row[1])]

      log("Computing the new embeddings...")
      new_embeddings = compute_new_embeddings(file_manager, model, ent_full_data)

    log(f"calling the write index method")
    write_index_file(file_manager, new_embeddings)

