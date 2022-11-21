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
import pickle

def write_index_file(fman):
    file = open(f"{fman.results_repo}/trained_blink_emb_11_22.tsv", "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

    log("reading embeddings from file...", 1)

    embeddings = []
    for row in reader:
        if len(row)>2:
          embeddings.append(str_to_tensor(row[1]))
        if len(embeddings) % 50000 == 0:
          log(f"processed {len(embeddings)} lines", 2)

    embeddings = np.array(embeddings, dtype=np.float32)

    log(f"indexing data ...", 1)
    index = DenseHNSWFlatIndexer(100, store_n=64, ef_search=32, ef_construction=32)
    index.index_data(embeddings)

    log(f"Writing index to file => {fman.results_repo}/trained_blink_emb_20_11")
    index.serialize(f"{fman.results_repo}/trained_blink_emb_20_11")

if __name__ == "__main__":
    file_manager = ModelFileIO()
    write_index_file(file_manager)


