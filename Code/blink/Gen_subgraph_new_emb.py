import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from Code.FileIOManager import ModelFileIO, time_format
from Code.blink.Dataset_processing import str_to_tensor
from Code.blink.biencoder import BiEncoderRanker
from Code.blink.indexer.faiss_indexer import DenseHNSWFlatIndexer
from Code.Logger import log
import requests
import faiss
import json
import re
import csv
import time
import threading
import os
import sys

sys.path.insert(0, os.getcwd())

def camel_case_split(s):
    idx = list(map(str.isupper, s))
    # mark change of case
    l = [0]
    for (i, (x, y)) in enumerate(zip(idx, idx[1:])):
        if x and not y:  # "Ul"
            l.append(i)
        elif not x and y:  # "lU"
            l.append(i+1)
    l.append(len(s))
    # for "lUl", index of "U" will pop twice, have to filter that
    s = ""
    for c in [s[x:y] for x, y in zip(l, l[1:]) if x < y]:
      s += c
    return s


def load_entity_full_description(f_man):
    labels_dict = {}

    with open(f"{f_man.repo}/DBpedia dumps/labels_lang=en.ttl", "r", encoding="utf-8") as l_file:
        line = "a"
        lid = 0
        while line:
            lid += 1
            line = l_file.readline()
            regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<label>.*)"@en''')
            m = regex.search(line)

            if m:
                labels_dict[m.group("link")] = camel_case_split(m.group("label"))

            if lid % 1000000 == 0:
                print(f"read {lid} line...")
#            if lid == 100000:
#                break

    print(f"labels ====> {len(labels_dict)}")

    with open(f"{f_man.repo}/DBpedia dumps/short-abstracts_lang=en.ttl", "r", encoding="utf-8") as l_file:
        line = "a"
        lid = 0
        found = 0
        wlabel = 0
        while line:
            lid += 1
            line = l_file.readline()
            regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<abstract>.*)"@en''')
            m = regex.search(line)

            if m:
                if m.group("link") in labels_dict.keys():
                    wlabel += 1
                    found += 1
                    labels_dict[m.group("link")] = (labels_dict[m.group("link")], m.group("abstract"))
                else:
                    label = m.group("link")[m.group("link").rfind("/") + 1:].replace("_", " ")
                    labels_dict[m.group("link")] = (label, m.group("abstract"))

            if lid % 1000000 == 0:
                print(f"read {lid} lines and found desc for {found} entities...")
#            if found == 1000:
#               break
    print(f"after abstracts ====> {len(labels_dict)},  ===> {wlabel} have label entry")
    return labels_dict


def gen_url_map(lines):
    global url_map
    positives = 0
    negatives = 0

    for line in lines:
      if "http://dbpedia.org" not in line:
        negatives += 1
      else:
        positives += 1
        line = line.split("\t")
        url_map.append(line[0])
    print(f"Finished mapping with {positives} positive and {negatives} negatives")

def get_aida_files_urls(file_manager):
    file_types = [
        "training",
        "validation",
        "evaluation"
    ]

    file_manager.add_path("training", "aida_train", "Datasets")
    file_manager.add_path("validation", "aida_testa", "Datasets")
    file_manager.add_path("evaluation", "aida_testb", "Datasets")

    entities_urls = []

    for ftype in file_types:
        t, e, m = file_manager.read_aida_input_doc(ftype)

        for key in list(e.keys()):
            entities_urls += e[key]

    return entities_urls


def read_embeddings(lines, t):
    if t == 1:
        global first_half
    else:
        global second_half

    skipped = 0
    start = time.time()
    for lid in range(len(lines)):
        line = lines[lid].split("\t")

        if not len(line) > 50:
          skipped+=1
          continue
        vec = [float(x.strip("\n\"")) for x in line[1:]]

        if t == 1:
            first_half.append(vec)
        else:
            second_half.append(vec)

        if lid % 1000000 == 0:
            # TODO: add ==> print( ... from {len(lines)}")
            end = time.time()
            print(f"Thread {t} has processed {lid} lines and skippend {skipped} from {len(lines)} in {time_format(int(end-start))}")
            start = time.time()

    start = time.time()
    if t == 1:
        first_half = np.array(first_half, dtype=np.float32)
    else:
        second_half = np.array(second_half, dtype=np.float32)
    end = time.time()

    print(f"it took thread_{t}  {time_format(int(end-start))} to convert to np arrays")


def load_transe_embeddings(f_man):
    """
    Read TransE embedding files, a tsv file, format: link \t e1 \t e2 ...\t e100, <====> ei:float
     :param f_man:
    :return:
    """
    file = open(f"{f_man.results_repo}/{fkge_name}", "r", encoding="utf-8")

    start = time.time()
    lines = file.readlines()
    end = time.time()

    print(f"It took {time_format(int(end - start))} to read the kge file")

    mapping_thread = threading.Thread(target=lambda: gen_url_map(lines))

    global url_map
    url_map = []

    mapping_thread.start()

    global first_half
    global second_half
    first_half = []
    second_half = []

    f_thread = threading.Thread(target=lambda: read_embeddings(lines[:len(lines) // 2], 1))
    s_thread = threading.Thread(target=lambda: read_embeddings(lines[len(lines)//2:], 2))

    start = time.time()
    f_thread.start()
    s_thread.start()

    f_thread.join()
    s_thread.join()
    end = time.time()

    print(f"It took {time_format(int(end - start))} to read the embeddings and extract them")

    start = time.time()
    complete_kge = np.concatenate((first_half, second_half), 0)
    end = time.time()
    print(f"It took {time_format(int(end - start))} to concatenate the results")

#    return complete_kge

#    index = DenseHNSWFlatIndexer(100)

#    start = time.time()
#    index.index_data(complete_kge)
#    end = time.time()

    print(f"It took {time_format(int(end - start))} to add data to the faiss index")
    mapping_thread.join()

    print(f"I have a map of size {len(url_map)} and a kge of size {complete_kge.shape}")
    return complete_kge
    #return index


def get_aida_files_urls(file_manager):
    file_types = [
        "training",
        "validation",
        "evaluation"
    ]

    file_manager.add_path("training", "aida_train", "Datasets")
    file_manager.add_path("validation", "aida_testa", "Datasets")
    file_manager.add_path("evaluation", "aida_testb", "Datasets")

    entities_urls = []

    for ftype in file_types:
        t, e, m = file_manager.read_aida_input_doc(ftype)

        for key in list(e.keys()):
            entities_urls += e[key]

    return entities_urls


def get_syntax_neighbors(index, syn_neighbors):
    global url_map
    neighbors = {}

    for url in range(len(syn_neighbors)):
        try:
            uid = url_map.index(url)
            ekge = index.reconstruct(uid)
        except ValueError:
            continue

        neighbors[url] = (ekge, syn_neighbors[url][0], syn_neighbors[url][1])

    return neighbors


def get_semantic_neighbors(index, data, emb):
    _, idx = index.search_knn(emb, 10)
    neighbors = {}

    for nid in idx:
        kge = index.reconstruct(nid)
        url = url_map[nid]

        neighbors[url] = (kge, data[url][0], data[url][1])

    return neighbors


def compile_dataset(f_index, data, aida_urls):
    """
    Organizes entities with their respective URL / KGE / LABEL / DESC
    and for each entity retrieve neighbors syntactical and semantical
    :param f_index:
    :param data:
    :param aida_urls:
    :return:
    """
    complete_dataset = {}
    global url_map

    for url in aida_urls:
        if url in data.keys():
            data_uid = list(data.keys()).index(url)
            try:
                uid = url_map.index(url)
                ekge = f_index.reconstruct(uid)
            except ValueError:
                continue

            complete_dataset[url] = (ekge, data[url][0], data[url][1])
            syn_neighbors = get_syntax_neighbors(f_index, data[data_uid - 5: data_uid + 5])
            sem_neighbors = get_semantic_neighbors(f_index, data, ekge)

            complete_dataset = {**complete_dataset, **sem_neighbors, **syn_neighbors}

    return complete_dataset


def compute_new_embeddings(fman, model, data, kge, t=None):
#def compute_new_embeddings(fman, data, kge, t=None):
    from Code.blink.data_process import get_candidate_representation
    global url_map

    new_embeddings = {}

    print(f"Thread_{t}: Got KGE of size {kge.shape}")
    print(f"Thread_{t}: {kge[0]}")
    print(f"Thread_{t}: {kge[0].shape}")

    if t == 1:
      print("opened file in {fman.results_repo} from thread 1")
      out_file = open(f"{fman.results_repo}/new_entries.tsv", "w", encoding="utf-8")
      map_file = open(f"{fman.results_repo}/links_map.txt", "w", encoding="utf-8")
    elif t == 2:
      print("opened file in {fman.results_repo} from thread 2")
      out_file = open(f"{fman.results_repo}/new_entries2.tsv", "w", encoding="utf-8")
      map_file = open(f"{fman.results_repo}/links_map_2.txt", "w", encoding="utf-8")

    writer = csv.writer(out_file, delimiter="\t")

    kge_batch = None
    token_ids_batch = None
    url_batch = []

    start = time.time()
    print(f" Iterating over {len(kge)} URLS")
    print(f"len(kge) > len(data) {len(kge) > len(data)}")

    offset = len(kge) if t ==2 else 0
    print(f"Thread_{t}: offset = {offset}")
    found = 0
    not_found = 0
    x = len(kge)
    url = []

    for url_id in range(x):

#        start = time.time()
#        print("started the loop at this point ", end=": ")

        if url_id % 500000 == 0:
          end = time.time()
          print(f"We processed {url_id} line from {x} and took {time_format(int(end-start))} found {url_id-found} links and skipped {not_found}")
          start = time.time()

        url = url_map[url_id + offset]

        if data.get(url):
            found += 1
            url_batch.append(url)
            map_file.write(f"{url}")

#            import pdb
#            pdb.set_trace()

            print(f"URL ====> {url} /////  data[url] =====> {data[url]}")

            entry = data[url]
            kg_entry = kge[url_id]

            if type(entry) != str:
              desc_token_ids = get_candidate_representation(entry[1], model.tokenizer, 196, entry[0])["ids"]
              desc_token_ids = torch.IntTensor(desc_token_ids).unsqueeze(0)
            else:
              desc_token_ids = get_candidate_representation("", model.tokenizer, 196, entry)["ids"]
              desc_token_ids = torch.IntTensor(desc_token_ids).unsqueeze(0)
#              desc_token_ids = desc_token_ids
#              if found == 1:
#                print(f"One example of desc_token_ids =={type(desc_token_ids)}=== {desc_token_ids} ")

#            if token_ids_batch is not None:
#            token_ids_batch = torch.cat((token_ids_batch, desc_token_ids), dim=0)
#            else:
#                token_ids_batch = desc_token_ids

#            if kge_batch is not None:
#                kge_batch = torch.cat((kge_batch, torch.from_numpy(kg_entry).unsqueeze(0)), dim=0)
#            else:
#                kge_batch = torch.from_numpy(kg_entry).unsqueeze(0)
            kge_batch = kg_entry
#            if found == 1:
#              print(f"One example of kge_batch ===={type(kge_batch)}=== {kge_batch} ")

            try:
              data.pop(url)
            except KeyError:
              print(f"did not find when popping {url}")
            writer.writerow([url, desc_token_ids, kge_batch])
        else:
          not_found += 1
#          print(f"We Lost {url}")

#    print(f"wrote {found} lines in the end")
#            if kge_batch.size(0) == 256:

#                print(f"entered first batch ")
#                out = model.encode_candidate(token_ids_batch, kge_emb=kge_batch)


#                print(f"Encoded the candidates to a vector of size ({out.size()}) ==> out[idx] = {out[1].size()}")

#                print(f"iterating over the url batch taht contains {len(url_batch)} links")
#                for u in range(len(url_batch)):
#                    new_embeddings[url_batch[u]] = out[u]
#                    writer.writerow([url_batch[u], out[u]])
#                print(f"new embeddings got new entries and now is of size ({len(new_embeddings)})")

#                end = time.time()
#                print(
#                    f"from {t}: kge_batch = {kge_batch.size()} // token_ids_batch"
#                    f" = {token_ids_batch.size()} --- time/batch = {time_format(int(end-start))}"
#                )

#                kge_batch = None
#                token_ids_batch = None
#                url_batch = []

#                end = time.time()
#                print(f"Finished processing the batch : {time_format(int(end-start))}")
#                start = time.time()

#                return None
#    print(f"returning from {t} with {len(new_embeddings)}")
#    return new_embeddings


def main():
    log("Reading the arguments...", 0)
    # model_name = "/pytorch_model.bin"
    model_name = "BLINK_output/pytorch_model.bin"
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
    reranker.load_model(f"{models_path}/{model_name}")

    #    state_dict = torch.load(f"{models_path}{model_name}")
    #    reranker.model.load_state_dict(state_dict)

    ent_txt_desc = load_entity_full_description(file_manager)
    # aida_urls = get_aida_files_urls(file_manager)
    kge_index = load_transe_embeddings(file_manager)

    # complete_dataset = compile_dataset(kge_index, ent_txt_desc, aida_urls)

    t1 = threading.Thread(target=lambda: compute_new_embeddings(file_manager, reranker, ent_txt_desc, kge_index[:len(kge_index)//2], 2))
    t2 = threading.Thread(target=lambda: compute_new_embeddings(file_manager, reranker, ent_txt_desc, kge_index[len(kge_index)//2:], 1))

#    t1 = threading.Thread(target=lambda: compute_new_embeddings(file_manager, ent_txt_desc, kge_index[:len(kge_index)//2], 2))
#    t2 = threading.Thread(target=lambda: compute_new_embeddings(file_manager, ent_txt_desc, kge_index[len(kge_index)//2:], 1))

    t2.start()
    t1.start()

    # file = open(f"{file_manager.results_repo}/dataset_desc_and_emb.tsv", "w", encoding="utf-8")
    # writer = csv.writer(file, delimiter="\t")
    #
    # for k in complete_dataset:
    #     writer.writerow([k, complete_dataset[k][0], complete_dataset[k][1], complete_dataset[k][2]])
    #
    # for key, val in .items():
    #     writer.writerow([key, val])


if __name__ == "__main__":
    # fkge_name = "entity_embeddings_dbp21-03_transe_dot.tsv"
    fkge_name = "sub_sub_kge.tsv"

    main()
