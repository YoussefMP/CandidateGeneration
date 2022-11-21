import os
import sys

sys.path.insert(0, os.getcwd())

from Code.Data_Manager import EmbeddingsMyDataset
from Code.FileIOManager import ModelFileIO
from Code.blink.Dataset_processing import read_complete_entity_dataset
from torch.utils.data import DataLoader
from Code.Logger import log
from Code.blink.blink_Core import load_params, load_model
from Code.blink.data_process import get_candidate_representation
from Code.blink.indexer.faiss_indexer import DenseHNSWFlatIndexer
import torch.nn as nn
import numpy as np
import pickle
import torch
import csv


test_file = "b"
NB_CANDS = 50


class Dataset:
    def __init__(self, mc, eg):
        self.x = []
        self.y = []

        for entry in range(len(mc)):
            self.x.append(torch.IntTensor(mc[entry]))
            self.y.append(eg[entry])

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def read_new_embeddings(fman):
    embeddings = []

    log(f"reading the entities embeddings...", 2)
    file = open(f"{fman.results_repo}/init_blink_emb_20_11_22.tsv", "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

#    urls_file = open(f"{fman.results_repo}/url_map_19_11.txt", "w", encoding="utf-8")

    c = 0
    for row in reader:
      c += 1
#      urls_file.write(f"{row[0]}\n")
      embeddings.append([float(e) for e in row[1:]])

      if c % 50000 == 0:
          log(f"Done with {c} entities", 3)

#    urls_file.close()

    HNSW = DenseHNSWFlatIndexer(1024, store_n=64, ef_search=32, ef_construction=32)

    log(f"Adding data to the index ...", 2)
    data_vecs = np.array(embeddings, dtype=np.float32)
    HNSW.index_data(data_vecs)
    HNSW.serialize(f"{fman.results_repo}/init_blink_emb_20_11_22")

    return HNSW, row


def read_final_file(fman, model, url_map):
  f = open(f"{fman.results_repo}/blink_Eval_pairs_{test_file}.tsv", "r", encoding="utf-8")

  reader = csv.reader(f, delimiter="\t")

  mentions_context = []
  eg = []

  t_count = 0
  p_count = 0

  for row in reader:

      t_count += 1
      if row[0] not in url_map:
          continue

      mentions_context.append([int(e) for e in row[1:]])
      eg.append(row[0])
      p_count += 1

  dataset = Dataset(mentions_context, eg)
  dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2)

  log(f"From {t_count} pairs we only_found {p_count} entities", 3)

  return dataloader


def get_mention_url_pairs(fman, model, url_map):
    """

    :param fm:
    :return:
    """
    if os.path.exists(f"{fman.results_repo}/blink_Eval_pairs_{test_file}.tsv"):
        return read_final_file(fman, model, url_map)

#    fman.add_path("test", f"aida_test{test_file}", extension="Datasets")
#    texts, eg_dict, mentions_dict, = fman.read_aida_input_doc("test")

    from Code.blink.Dataset_processing import get_mentions_entities_dataset
    mentions_context, entities = get_mentions_entities_dataset(fman, f"aida_test{test_file}", "test", model.tokenizer, testing=True)

    log(f"Setting the dataloader with the test data...{len(list(mentions_context.values()))}", 0)
    dataloader = EmbeddingsMyDataset(list(mentions_context.values()), list(entities.values()), model.tokenizer, testing=True)
    log(f"Loaded pairs and entities => size of the dataset {len(dataloader)}", 1)
    train_dataloader = DataLoader(dataset=dataloader, batch_size=8, shuffle=True, num_workers=2)

    fo = open(f"{fman.results_repo}/blink_Eval_pairs_{test_file}.tsv", "w", encoding="utf-8")
    writer = csv.writer(fo, delimiter="\t")

    for id, data in enumerate(dataloader):
      writer.writerow([data[1]] + data[0].tolist())

    return dataloader



def eval_model(encoder, dataloader, f_idx, url_map, set_size, fman):
    log("Starting model's evaluation...", 1)

    # Loss function and optimizer
    criterion = nn.CosineSimilarity()

    # Training Loop
    encoder.eval()

    tp_count = 0
    total_count = 0

    max_cos = 0
    printed_max = 0

    batch_cosine_sum = []

    per_ent_found = {}

    # TODO: - Initialize DS for the data to be written to file.
    #       -
    for idx, data in enumerate(dataloader):

        eg = data[1]
        source = data[0].to(encoder.device)

        target = []
        for url in eg:
          uid = url_map.index(url)
          t_vec = f_idx.index.reconstruct(uid)[:-1]
          target.append(torch.FloatTensor(t_vec))

        with torch.no_grad():

            # outputs = encoder(source, target)
            outputs = encoder.encode_context(source).cpu().detach()            # New model evaluation input

            D, I = f_idx.search_knn(outputs.numpy(), set_size)

        for lid in range(len(I)):           # lid = list id
            max_batch_cos = []
            total_count += 1

            if eg[lid] in per_ent_found.keys():
                per_ent_found[eg[lid]][0] += 1
            else:
                per_ent_found[eg[lid]] = [1, 0]

            for cid in I[lid]:              # cid = candidate id
                cid = int(cid)
                pe = url_map[cid]                             # pe = predicted entity
                pv = f_idx.index.reconstruct(cid)[:-1]             # pv = predicted vector

                max_batch_cos.append(criterion(torch.FloatTensor(pv).unsqueeze(0), target[lid]).unsqueeze(0))
                batch_cosine_sum.append(criterion(torch.FloatTensor(pv).unsqueeze(0), target[lid]).unsqueeze(0))

                if criterion(target[lid], torch.FloatTensor(outputs[lid]).unsqueeze(0)) > max_cos:
                    max_cos = criterion(target[lid], torch.FloatTensor(outputs[lid]).unsqueeze(0))
                    eg_max = pe
                    predicting = eg[lid]

                if pe == eg[lid]:
                    tp_count += 1
                    log(f"Found one more {eg[lid]} ==> CosSim(prediction, golden_entity) ="
                        f" {criterion(target[lid], torch.FloatTensor(outputs[lid]).unsqueeze(0))}", 2)
                    if eg[lid] in per_ent_found.keys():
                        per_ent_found[eg[lid]][1] += 1
                    else:
                        per_ent_found[eg[lid]] = [0, 1]
                    break

            if max_cos > printed_max:
                log(f"Highest Cos sim achieved ==> {max_cos} for eg = {eg_max} ===> while predicting {predicting}", 3)
                printed_max = max_cos

            # log(f"Closest predicted link is {url_map[I[lid][max_batch_cos.index(max(max_batch_cos))]]} ==> EG = {eg[lid]}", 4)

        if idx % 10 == 0:
            log(f"We are {(idx * 100) / dataloader.__len__()}% Done,", 1)

    avg_cs_batch = sum(batch_cosine_sum) / len(batch_cosine_sum)
    gold_recall = tp_count / total_count
    log(f"GOLD Recall {gold_recall} ===> {tp_count} / {total_count} --- avg cosine score per batch = {avg_cs_batch}")

    if not os.path.exists(f"{fman.models_repo}/BLINK_output/Eval"):
        try:
            os.mkdir(f"{fman.models_repo}/BLINK_output/Eval/")
        except:
            pass
    with open(f"{fman.models_repo}/BLINK_output/Eval/init_results_{test_file}.json", "w", encoding="utf-8") as of:
        of.write(f"File={test_file}\n")
        of.write(f"candidates_retrieved={NB_CANDS}\n")
        of.write(f"average_cosine_sim_per_batch={avg_cs_batch}\n")
        of.write(f"max_cosine_sim_score={printed_max}\n")
        of.write(f"gold_recall={gold_recall}\n")
        of.write(f"true_positives={tp_count}\n")

    print()
    print()
    for k in per_ent_found:
        if per_ent_found[k][1] > 0:
            print(f"{k} ===> {per_ent_found[k]}")

    return gold_recall


if __name__ == "__main__":
  model_name = "biencoder_wiki_large.bin"

  file_manager = ModelFileIO()

  if not os.path.exists(f"{file_manager.results_repo}/init_blink_emb_20_11_22"):
      log(f"Reading the file for the entity_data...", 1)
      HNSW, row = read_new_embeddings(file_manager)
  else:

      log(f"Loading index from file...", 1)
      HNSW = DenseHNSWFlatIndexer(1024, store_n=64, ef_search=64, ef_construction=32)
      HNSW.deserialize_from(f"{file_manager.results_repo}/init_blink_emb_20_11_22")

  urls_file = open(f"{file_manager.results_repo}/init_blink_url_map_20_11_22.pkl", "rb")
  url_map = pickle.load(urls_file)



  log(f"Loading the model....", 1)
  args = {"batchsize": 8, "index_emb_size": 100}

  params, model_path = load_params(file_manager, "", args)
  reranker = load_model(params, f"{model_path}/", model_name)

  log(f"Loading evaluation data...", 1)
  dataloader = get_mention_url_pairs(file_manager, reranker, url_map)

  eval_model(reranker, dataloader, HNSW, url_map, NB_CANDS, file_manager)
