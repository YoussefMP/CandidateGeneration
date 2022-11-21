import os
import sys

sys.path.insert(0, os.getcwd())

import csv
import re
import os
import torch
import pickle
import faiss
import numpy as np
from Code.FileIOManager import ModelFileIO
from Code.blink.biencoder import BiEncoderRanker
from Code.blink.data_process import get_candidate_representation

from GPUtil import showUtilization as gpu_usage
from numba import cuda

def load_data_from_dumpbs():
    """
    Read the DBpedia dump file and extract labels and descriptions of the entities present in the dataset
    :return: complete_entities_dataset {txt_id: [{link: str: DBpedia_link,
                                                 label: str: entity label,
                                                 abstract: str: entity description
                                                }]
                                        }
    """
    criterions = ["label", "abstract"]
    complete_entities_dataset = {}

    for criteria in criterions:
        file_path = f"{file_manager.repo}\\DBpedia dumps\\"
        if criteria == "label":
            file_path += "labels_lang=en.ttl"
            regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<label>.*)"@en''')
        else:
            file_path += "short-abstracts_lang=en.ttl"
            regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<abstract>.*)"@en''')

        with open(
                file_path,
                "r", encoding="utf-8") as file:
            line = "a"
            t = 0
            while line:
                t += 1
                line = file.readline()
                m = regex.match(line)

                if m:
                    if criteria == "label":
                        complete_entities_dataset[m.group("link")] = [m.group("label")]
                    else:
                        try:
                            complete_entities_dataset[m.group("link")].append(m.group("abstract"))
                        except KeyError:
                            label = m.group("link")[m.group("link").rfind("/")+1:].replace("_", " ")
                            complete_entities_dataset[m.group("link")] = [label, m.group("abstract")]

        file.close()
    return complete_entities_dataset


def load_data_from_file():
    import csv
    file = open(f"{file_manager.results_repo}database.tsv", "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

    entity_dataset = {}
    count = 0

    for row in reader:
        if len(row) > 0:
            count += 1
            entity_dataset[row[0]] = [row[1], row[2]]
        if count % 1000000 == 0:
          print(f"Processed {count} rows")
        # if count == 100000:
        #   break
    return entity_dataset


def load_params():
    import json
    config_name = model_name.replace(".bin", ".json")
    models_path = file_manager.models_repo
    with open(f"{models_path}/{config_name}") as j_file:
        params = json.load(j_file)
    params["kge_emb_dim"] = 0

    return params, models_path


def load_model():
    model = BiEncoderRanker(param)
    model.load_model(f"{model_path}/{model_name}")

    state_dict = torch.load(f"{model_path}/{model_name}")
    model.model.load_state_dict(state_dict)
    model.add_layer()

    model.to(device)

    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Before emptying cache")
gpu_usage()

torch.cuda.empty_cache()
cuda.select_device(0)
cuda.close()
cuda.select_device(0)

print("After emptying cache")
gpu_usage()

model_name = "biencoder_wiki_large.bin"
file_manager = ModelFileIO("path", "shallom")

complete_dataset = load_data_from_file()

# Reading the json file and adding the parameters for training
param, model_path = load_params()

# Instantiating the bi-encoder model for training
re_ranker = load_model()
tokenizer = re_ranker.tokenizer

entity_id_map = {}
batch = []
complete_tensor = None
b_count = 0
ent_id = 0

for k, data_item in complete_dataset.items():
    entity_id_map[ent_id] = k
    ent_id += 1

    label = data_item[0]
    if len(data_item) == 2:
        desc = data_item[1]
    else:
        desc = "NaN"

    ent_rep = get_candidate_representation(desc, tokenizer, 196, label)
    batch.append(ent_rep["ids"])

    if len(batch) == 256 or len(entity_id_map) == len(list(complete_dataset.keys())):
        b_count += 1
        batch_input = torch.IntTensor(batch).to(device)
        if len(batch_input.size()) > 2:
            batch_input = batch_input.squeeze(1)
        output = re_ranker.encode_candidate(batch_input)

        if complete_tensor is None:
            complete_tensor = output
        else:
            complete_tensor = torch.cat((complete_tensor, output), dim=0)

        batch = []
        print(f"Processed {b_count} batches => {b_count * 256} entities / {len(list(complete_dataset.keys()))}")


final_array = complete_tensor.numpy()

index = faiss.IndexHNSWFlat(1024, 16)
index.hnsw.efSearch = 16
index.hnsw.efConstruction = 32

faiss.normalize_L2(final_array)
index.add(final_array)

faiss.write_index(index, f"{file_manager.repo}/Datasets/my_index.index")

with open(f"{file_manager.repo}/Datasets/index_map.sav", "wb") as dump_file:
  pickle.dump(entity_id_map, dump_file, protocol=pickle.HIGHEST_PROTOCOL)
