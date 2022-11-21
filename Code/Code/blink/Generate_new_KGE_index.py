import os
import sys

sys.path.insert(0, os.getcwd())

from Code.blink.indexer.faiss_indexer import DenseHNSWFlatIndexer
import numpy as np
import torch
from Code.FileIOManager import ModelFileIO, time_format
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
import csv
import time
import threading


def get_entities_embeddings(f_man, size):
    import csv
    import os

    index = {}
    file_path = os.path.join(f_man.repo, "Results/complete_entity_dataset.tsv")

    file = open(file_path, "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

    for row in reader:
        if len(row) > 1:
            index[row[0]] = [f"[CLS]{row[2]}[ENT]{row[3]}[SEP]", str_to_tensor(row[1])]
            if len(index) == size and size != 0:
                break

    return index


def generate_new_embeddings(f_man, args, model=None):
    index_name = args["indexname"]
    kge_embeddings = get_kge_index_embeddings(f_man, index_name, 10)

    new_embeddings = np.empty((0, args["index_embedding_size"]), dtype=np.float32)

    # TODO: remove the if condition after finishing testing
    if model:
        for k, v in kge_embeddings.items():
            tokenized_cand = model.tokenizer.tokenize(v[0])
            cand_ids = torch.IntTensor(model.tokenizer.convert_tokens_to_ids(tokenized_cand)).unsqueeze(0)
            kge_emb = torch.FloatTensor(v[1]).unsqueeze(0)

            entity_embedding = model.encode_candidate(cand_ids, kge_emb)

            new_embeddings = np.append(new_embeddings, np.array(entity_embedding), axis=0)

        faiss.normalize_L2(new_embeddings)

    k_probing = 2

    with open(f"{f_man.repo}/Results/{index_name}_new_embeddings.npy", "wb") as out_file:
        np.save(out_file, new_embeddings)
    out_file.close()

    index = init_and_train_index(args["index_embedding_size"], cells, new_embeddings)
    index.nprobe = k_probing

    # print(index.ntotal)
    #
    # t = torch.rand((1, 25))
    #
    # k =5
    # D, I = index.search(np.array(t), k)
    #
    # print(D, I)
    # print()


# def load_index(args):
#     index_name = args["indexname"]
#     new_embeddings_path = f"{file_manager.repo}\\Results\\{index_name}_new_embeddings.npy"
#
#     with open(new_embeddings_path, "rb") as d_file:
#         new_embeddings = np.load(d_file)
#     d_file.close()
#
#     dim = new_embeddings[0].shape[-1]
#     index = init_and_train_index(dim, cells, new_embeddings)
#
#     return index

def prepare_faiss_index(file_manager, index_name, nb_cells):
    """

    :param file_manager:
    :param index_name:
    :param nb_cells:
    :return:  entity_embeddings = {key=url: value=embedding}
    """
    if "shallom" in index_name:
        file_name = "Shallom_entity_embeddings.csv"
    elif "transe" in index_name:
        file_name = "transe_entity_embeddings.tsv"
    else:
        file_name = "error"

    # indexed_embeddigns = {}
    entity_embeddings = file_manager.read_csv_embeddings_file(file_name)
    embeddings = np.array(list(entity_embeddings.values()), dtype=np.float32)

    # eid = 0
    # for k, v in entity_embeddings.items():
    #     indexed_embeddigns[eid] = (k, v)
    #     eid += 1

    dim = embeddings.shape[1]

    quantizer = faiss.IndexFlatIP(dim)
    faiss_index = faiss.IndexIVFFlat(quantizer, dim, nb_cells)

    if not faiss_index.is_trained:
        faiss_index.train(embeddings)

    faiss_index.add(embeddings)

    return entity_embeddings, faiss_index


def load_entity_full_description(f_man, links):

    print("Loading labels....")

    complete_dict = {}

    with open(f"{f_man.repo}/DBpedia dumps/labels_lang=en.ttl", "r", encoding="utf-8") as l_file:
        line = "a"
        lid = 0
        found = 0
        while line:

            lid += 1
            line = l_file.readline()

            if "dbpedia" not in line:
                continue

            regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<label>.*)"@en''')
            m = regex.search(line)

            if m:
                if m.group("link") in links.keys():
                    complete_dict[m.group("link")] = [links[m.group("link")], m.group("label")]
                    found += 1

            if len(links) == found:
                print("Found All entities labels....")

            if found % 100000 == 0:
                print(f"Found ===> {found}")

            if lid % 500000 == 0:
                print(f"Making progress {lid}")

    print(f"Found  ======> {found} labels")
    with open(f"{f_man.repo}/DBpedia dumps/short-abstracts_lang=en.ttl", "r", encoding="utf-8") as l_file:
        line = "a"
        lid = 0
        found = 0
        ml = 0
        while line:

            lid += 1
            line = l_file.readline()

            if "dbpedia" not in line:
                continue

            regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<abstract>.*)"@en''')
            m = regex.search(line)

            if m:
                if m.group("link") in complete_dict.keys():
                    found += 1
                    complete_dict[m.group("link")].append(m.group("abstract"))
                elif m.group("link") in links.keys():
                    reg = re.compile(r''' ?\(.*\) ?''')
                    label = m.group("link")[m.group("link").rfind("/")+1:].replace("_", "")
                    label = re.sub(reg, "", label)
                    complete_dict[m.group("link")] = [links[m.group("link")], label, m.group("abstract")]
                    found += 1
                    ml += 1

            if lid % 500000 == 0 or found % 100000 == 0:
                print(f"read {lid} lines and found desc for {found} entities from which {ml}...")

    fo = open(f"{f_man.results_repo}/Complete_17_11_22.tsv", "w", encoding="utf-8")
    writer = csv.writer(fo, delimiter="\t")

    for key in complete_dict:
        if len(complete_dict[key]) == 3:
            writer.writerow([key, complete_dict[key][0], complete_dict[key][1], complete_dict[key][2]])
        else:
            writer.writerow([key, complete_dict[key][0], complete_dict[key][1], ""])

    return complete_dict

def load_transe_embeddings(f_man):
    embeddings = {}

    file = open(f"{f_man.results_repo}/sub_kge_18_11_22.tsv", "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

    for row in reader:
        vector = [float(i) for i in row[1:]]
        embeddings[row[0]] = torch.FloatTensor(vector)

    return embeddings

def load_ent_data(fman):
    file = open(f"{fman.results_repo}/Complete_17_11_22.tsv", "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

    data = {}

    for row in reader:
        if len(row) > 2:
            data[row[0]] = [str_to_tensor(row[1]), row[2], row[3]]

    return data


def compute_new_embeddings(fman, model, data, t=None):
    from Code.blink.data_process import get_candidate_representation

    file = open(f"{fman.results_repo}/new_embeddings_18_11_22.tsv", "w", encoding="utf-8")
    writer = csv.writer(file, delimiter="\t")

    url_map_file = open(f"{fman.results_repo}/url_map_18_11_22.txt", "w", encoding="utf-8")

    new_embeddings = {}

    kge_batch = None
    token_ids_batch = None
    url_batch = []

    bp = 0

    start = time.time()
    for url in data:

        abstract = data[url][2]
        label = data[url][1]
        emb = torch.FloatTensor(data[url][0])

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

        if kge_batch.size(0) == 16:
            bp += 1

            out = model.encode_candidate(token_ids_batch.to(model.device), kge_emb=kge_batch.to(model.device))

            out.cpu().detach()
            token_ids_batch.cpu().detach()
            kge_batch.cpu().detach()

            if bp % 100 == 0:
                log(f"Encoded the candidates to a vector of size ({out.size()}) ==> out[idx] = {out[1].size()}", 1)

                log(f"iterating over the url batch taht contains {len(url_batch)} links", 1)
            for u in range(len(url_batch)):
                url_map_file.write(f"{url_batch[u]}\n")
                writer.writerow([url_batch[u]]+out[u].tolist())
                if bp ==1:
                  log(f"Writing to file {url_batch[u]} , {len(out[u].tolist())} //// len(row)={len([url_batch[u]]+out[u].tolist())}", 3)
#                new_embeddings[url_batch[u]] = out[u]
            if bp % 100 == 0:
                log(f"new embeddings got new entries ===> batchs processed = ({bp})", 1)

            end = time.time()
            if bp % 100 == 0:
                log(f"from {t}: kge_batch = {kge_batch.size()} // token_ids_batch = {token_ids_batch.size()} --- time/batch = {time_format(int(end - start))}", 1)

            kge_batch = None
            token_ids_batch = None
            url_batch = []

    print(f"returning from {t} with {len(new_embeddings)}")
    return new_embeddings


def write_new_emb_to_file(fman, data):

    file = open("{fman.results_repo}/new_embeddings_18_11_22.tsv", "r", encoding="utf-8")
    writer = csv.writer(file, delimiter="\t")

    url_map_file = open(f"{fman.results_repo}/url_map_18_11_22.txt", "w", encoding="utf-8")
    npdata = np.array([])

    log(f"writing the data to the csv file", 1)
    for url in data:
        url_map_file.write(f"{url}\n")
        np.append(npdata, data[url].numpy())
        writer.writerow([url, data[url]])

    log(f"Building the faiss index")
    d = 100  # Dimensionality of our vectors
    index = DenseHNSWFlatIndexer(d, store_n=64, ef_search=64, ef_construction=64)
    index.index_data(npdata)

    log("index constructed, now saving to file", 1)
    faiss.write_index(index, f"{fman.results_repo}/index_18_11_22")

def main():
    log("Reading the arguments...", 0)
    model_name = "pytorch_model.bin"
    kge_emb_dim = 100

    log("Instantiating the FileIO instance", 0)
    file_manager = ModelFileIO()

    log("Reading config file of the pretrained model", 0)
    import json
    config_name = "/biencoder_wiki_large.json"
    models_path = file_manager.models_repo

    with open(f"{models_path}{config_name}") as j_file:
        params = json.load(j_file)
        params["kge_emb_dim"] = kge_emb_dim
    j_file.close()

    reranker = BiEncoderRanker(params)
    log("Loading the model's weights...", 0)
    reranker.load_model(f"{models_path}/BLINK_output/{model_name}")

#    print("reading transe embeddings ...")
#    transe_emb = load_transe_embeddings(file_manager)
#    print("finished loading embeddings")
#    ent_txt_desc = load_entity_full_description(file_manager, transe_emb)

    log("Loading entities with desc and embedding...")
    ent_full_data = load_ent_data(file_manager)
    log("Computing the new embeddings...")
    new_embeddings = compute_new_embeddings(file_manager, reranker, ent_full_data)
#    write_new_emb_to_file(file_manger, new_embeddings)


if __name__ == "__main__":
    main()
