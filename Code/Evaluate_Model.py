from FileIOManager import ModelFileIO, str_to_tensor, time_format
from Code.blink.indexer.faiss_indexer import DenseHNSWFlatIndexer
from Code.Attention_Encoder import initiate_model
from Code.Data_Manager import FinalDataset
from Code.MentionsEncoder import Encoder
from torch.utils.data import DataLoader
from Logger import log
import torch.nn as nn
import numpy as np
import os
import torch
import csv
import time
import faiss


__DEBUG__ = False

index_name = "test_index4"
url_map_name = "small_url_map4.txt"

test_file = "a"
NB_candidates = 50
model_name = "ALL_BCECOS_4context_lr6_CG10"

context = "4context"


# ###################### Helper function
def tensor_to_str(tensor):
    req_str = "["

    for val in tensor:
        req_str += str(val.item()) + ","

    req_str = req_str[:-1] + "]"
    return req_str


def tensor_to_list(tensor):
    req_list = []

    for val in tensor:
        req_list.append(val.item())

    return req_list
# ###########################################################


# ##################### Data extraction functions
def read_final_file(fman):
    file = open(f"{fman.results_repo}/{context}Eval_pairs_{test_file}.tsv", "r", encoding="utf=8")
    reader = csv.reader(file, delimiter="\t")

    entries = []
    for row in reader:
        if len(row) > 2:
            entries.append([row[0], row[1], row[2], str_to_tensor(row[3])])

            if len(entries) % 3000 == 0:
                log(f"processed {len(entries)} rows", 1)

    return entries, "From File"


def get_mention_url_pairs(fm):
    """

    :param fm:
    :return:
    """
    # if os.path.exists(f"{fman.results_repo}/{context}Eval_pairs_{test_file}.tsv"):
    #     return read_final_file(fman)

    fm.add_path("test", f"aida_test{test_file}", extension="Datasets")
    texts, eg_dict, mentions_dict, smc = fm.read_aida_input_doc("test")

    encoder = Encoder()
    with_context = True if "context" in context else False
    with_context = False if "nocontext" in context else with_context
    d, tc = encoder.embed_mentions(texts, eg_dict, mentions_dict, with_context)

    fo = open(f"{fman.results_repo}/{context}Eval_pairs_{test_file}.tsv", "w", encoding="utf-8")
    writer = csv.writer(fo, delimiter="\t")

    for val in d:
        writer.writerow([val[0], val[1], val[2], val[3]])

    fo.close()
    return d, tc


def prepare_training_data(dataset, kge, url_list):

    log(f"Preparing the data for training...", 1)
    start = time.time()
    mapped_data = []

    entered_pairs = 0
    process = 0
    cache_hit = 0

    cached_entries = {}

    log(f"{len(dataset)} mentions to find, We have {len(url_list)} links in the KGE", 2)

    for entry in dataset:
        process += 1

        if process % 3000 == 0 or process == len(dataset):

            log(f"From {len(dataset)} processed {process} and found {entered_pairs} URLS in the KGE from which "
                f"{cache_hit} are cache hits", 2)

        url = entry[1]
        m_emb = entry[3]

        if url not in cached_entries.keys():
            try:
                eid = url_list.index(url)
                entered_pairs += 1
            except ValueError:
                continue

            target_vec = kge.index.reconstruct(eid)[:-1]
            cached_entries[url] = target_vec

        else:
            target_vec = cached_entries[url]
            cache_hit += 1

        mapped_data.append([url, m_emb, target_vec])

        if entered_pairs == len(dataset):
            break

    dataset = FinalDataset(mapped_data)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=2)
    end = time.time()

    log(f"Time needed to load all data into memory ==> {time_format(int(end-start))}\n")
    log(f"From {len(dataset)} we extracted {entered_pairs + cache_hit} pairs that will be used for training")

    return dataloader
#############################################################


# ######################## Index preparation functions
def load_index_from_file(fman):

    url_map = []
    d = 100  # Dimensionality of our vectors
    index = DenseHNSWFlatIndexer(d, store_n=64, ef_search=64, ef_construction=64)

    s = time.time()
    index.index = faiss.read_index(f"{fman.results_repo}/{index_name}")
    e = time.time()
    log(f"{time_format(e-s)} for loading the index from file", 1)

    url_f = open(f"{fman.results_repo}/{url_map_name}", "r", encoding="utf-8")
    line = "a"
    k = 0
    s = time.time()
    while line:
        line = url_f.readline()
        url_map.append(line.strip("\n"))
    e = time.time()
    log(f"{e-s} is the time for reading the urls file", 2)
    return index, url_map


def load_transe_index(fman):
    """
    loads the TransE embeddings into a FAISS index
    :return:
    """
    if os.path.exists(f"{fman.results_repo}/{url_map_name}"):
        return load_index_from_file(fman)

    fman.add_path("index", "sub_sub_kge.tsv", "Datasets")

    log("Opening kge file...", 1)
    file = open(f"{fman.files['index']}", "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

    entries = 0
    url_map = []
    index_vector = []

    start = time.time()
    for row in reader:

        if len(row) > 100:
            entries += 1
            url_map.append(row[0])
            temp_vec = []

            for e in row[1:]:
                e = e.strip("\"\n")
                temp_vec.append(float(e))
        else:
            continue

        if entries % 500000 == 0:
            end = time.time()
            log(f"Processed {entries} rows in {time_format(int(end - start))}", 2)
            start = time.time()

        index_vector.append(temp_vec)

    index_start = time.time()
    d = 100  # Dimensionality of our vectors
    index = DenseHNSWFlatIndexer(d, store_n=64, ef_search=64, ef_construction=64)

    log("Adding data to index...", 1)
    data_vec = np.array(index_vector, dtype=np.float32)
    index.index_data(data_vec)
    index_end = time.time()

    log(f"It took {time_format(int(index_end-index_start))} to finish building the index...", 0)

    faiss.write_index(index.index, f"{fman.results_repo}/test_index")

    url_io = open(f"{fman.results_repo}/url_map.txt", "w", encoding="utf-8")
    for u in url_map:
        url_io.write(u + "\n")

    return index, url_map
###############################################################


def eval_model(encoder, dataloader, f_idx, url_map, set_size):
    # Training Hyper parameters
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

        eg = data[0]
        source = data[1]
        target = data[2]

        with torch.no_grad():

            entry_size = int(768 / encoder.source_size)
            source = source.reshape([source.size(0), entry_size, -1])
            source = source.mean(1)

            # outputs = encoder(source, target)
            outputs = encoder(source)[1]            # New model evaluation input

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

                # TODO: Breakpoint to see what is needed to be implemented here
                #           - CosSim of target (eg[lid]) and prediction (pv)
                #           - Recall@k
                #           - time
                #           - Precision ?

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

    if not os.path.exists(f"{fman.models_repo}\\{model_name}\\Eval"):
        os.mkdir(f"{fman.models_repo}\\{model_name}\\Eval\\")

    with open(f"{fman.models_repo}/{model_name}/Eval/results.json", "w", encoding="utf-8") as of:
        of.write(f"File={test_file}\n")
        of.write(f"candidates_retrieved={NB_candidates}\n")
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

    fman = ModelFileIO()
    log("Getting the data from the file", 0)
    data, text_chunks = get_mention_url_pairs(fman)

    log("Loading the TranE embeddings", 0)
    f_index, urls = load_transe_index(fman)

    log("Preparing the dataloader...", 0)
    dataloader = prepare_training_data(data, f_index, urls)

    default_params = {"source_size": 768, "target_size": 100, "nb_layers": 12,
                      "nb_heads": 12, "fwd_expansion": 4, "dp": 0.1}

    model = initiate_model(f"{fman.models_repo}/{model_name}/{model_name}.bin", **default_params)

    # from Code.QuickModel import QModel
    # model = QModel(768, 100)
    # model.load_state_dict(torch.load(f"{fman.models_repo}/{model_name}/{model_name}.bin"))

    eval_model(model, dataloader, f_index, urls, NB_candidates)
