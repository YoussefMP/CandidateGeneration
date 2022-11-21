# __________________________________________________________________________________________________
import json

from Code.blink.Dataset_processing import prepare_data
import os
import torch
from Code.blink.blink_Core import load_model, load_params
from Code.blink.train_biencoder import get_kge
from Code.blink.Dataset_processing import generate_ent_set, extend_entity_dataset
from Code.FileIOManager import ModelFileIO
import re
from Code.Logger import log
import csv


def generate_graph_context(lgc, agc):
    """

    :param lgc: label graph context {URL: label}
    :param agc: abstract grph context {url: abstract}
    :return:
    """

    keys = list(lgc.keys())
    graph_context = {}
    processed = []

    batch = []
    for key in keys:
        batch.append(key)
        if len(batch) == 10:
            kge = get_kge(batch, "transe")

            for k in kge:
                lgc[k] = (lgc[k], kge[k])
                processed.append(k)

    for key in agc.keys():
        if key in processed:
            continue
        else:
            label = key[key.rfind("/")+1:].replace("_", " ")
            kge = get_kge([key], "transe")

            for k in kge:
                lgc[k] = (label, kge[k])
    return lgc


def write_complete_entity_dataset(file_manager, labels, abstracts):
    """
    Write tsv file: Row Format = URL \t Embedding \t Label \t Description
    :param labels:
    :param abstracts:
    :return:
    """
    # TODO: Rename
    file_path = f"{file_manager.repo}/eval_complete_entity_dataset.tsv"

    file = open(file_path, "w", encoding='utf8')
    writer = csv.writer(file, delimiter="\t")

    entity_dataset = {}
    for entity in labels.keys():
        try:
            description = abstracts[entity]
            writer.writerow([entity, labels[entity][1], labels[entity][0], description])
            entity_dataset[entity] = (labels[entity][1], labels[entity][0], description)

        except KeyError:
            writer.writerow([entity, labels[entity][1], labels[entity][0], "NaN"])
            entity_dataset[entity] = (labels[entity][1], labels[entity][0], "NaN")
    file.close()

    return entity_dataset


def filter_entities_for_description(data_entities, criteria=None, contexts=None):
    """
    Read the DBpedia dump file and extract labels and descriptions of the entities present in the dataset
    :param data_entities:
    :param criteria
    :return: complete_entities_dataset {txt_id: [{link: str: DBpedia_link,
                                                 label: str: entity label,
                                                 abstract: str: entity description
                                                }]
                                        }
            graph_context  {entity_url: label}
    """

    file_path = f"{file_manager.repo}\\DBpedia dumps\\"

    if criteria == "label":
        file_path += "labels_lang=en.ttl"
        regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<label>.*)"@en''')
    else:
        file_path += "short-abstracts_lang=en.ttl"
        regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<abstract>.*)"@en''')

    entities = {}
    if criteria == "label":
        for d in data_entities:
            entities = {**entities, **d}
    else:
        entities = data_entities

    complete_entities_dataset = {}

    hit = False
    count = 0

    with open(
            file_path,
            "r", encoding="utf-8") as file:

        lines = file.readlines()
        t = 0

        for lid in range(len(lines)):
            line = lines[lid]

            t += 1
            m = regex.match(line)

            if m:

                if hit and criteria == "label":
                    for i in range(lid-15, lid+15, 10):
                        g_context = get_kge([regex.match(line2).group("link") for line2 in lines[i: i+10] if regex.match(line2)], "transe")
                        gc_labels = {regex.match(line2).group("link"): regex.match(line2).group("label") for line2 in lines[i: i+10]}

                        for k, v in g_context.items():
                            if v is not None:
                                complete_entities_dataset[k] = (gc_labels[k], v)
                        hit = False

                full_link = m.group("link") in entities.keys()
                partial_link = m.group("link").replace("http://dbpedia.org", "") in entities.keys()

                if full_link or partial_link:
                    count += 1
                    try:
                        key = m.group("link").replace("http://dbpedia.org", "") if partial_link else\
                            m.group("link") if full_link else None

                        if criteria == "label":
                            complete_entities_dataset[m.group("link")] =\
                                (m.group("label"),
                                 torch.FloatTensor(entities[key]))
                            hit = True
                        else:
                            complete_entities_dataset[key] = m.group("abstract")

                    except KeyError as k_err:
                        print(f"KeyError: {k_err} \n ")

            if t % 15000 == 0:
                log(f"Parsed = {t} lines, and found data for {count}/{len(entities.keys())} entities", 4)

            if len(entities.keys()) == count and criteria == "label":
                break
        log(f"Results of filtering entities with no {criteria} => {count} / {len(entities)} = {count/len(entities)}%", 3)

    file.close()
    return complete_entities_dataset


file_manager = ModelFileIO()

file_types = [
    "training",
    "validation",
    "evaluation"
]

file_manager.add_path("training", "aida_train", "Datasets")
file_manager.add_path("validation", "aida_testa", "Datasets")
file_manager.add_path("evaluation", "aida_testb", "Datasets")


entities_urls = []

kge = []

for ftype in file_types:
    t, e, m = file_manager.read_aida_input_doc(ftype)

    for key in list(e.keys())[:12]:
        entities_list = e[key]

        for i in range(10, len(entities_list), 10):
            batch = entities_list[i-10: i]
            kge.append(get_kge(batch, "transe"))

            if i+10 > len(entities_list):
                batch = entities_list[i:]
                kge.append(get_kge(batch, "transe"))


if os.path.exists(f"{file_manager.results_repo}/eval_complete_entity_dataset.tsv"):
    # TODO: Remove eval_ form file name
    entities_data = read_complete_entity_dataset(f"{file_manager.results_repo}/complete_entity_dataset.tsv")
else:
    # # Get the entities for which we have a label and a description
    entity_label_dict = filter_entities_for_description(kge, criteria="label")
    entity_abstract_dict = filter_entities_for_description(entity_label_dict, criteria="abstract")
    entities_data = write_complete_entity_dataset(file_manager, entity_label_dict, entity_abstract_dict)


model_name = ""
args = {"batchsize": 8, "index_emb_size": 100}
params, model_path = load_params(file_manager, model_name, args)
reranker = load_model(params, model_path, model_name)
tokenizer = reranker.tokenizer

log(f"Extending the entity dataset with [CLS]label[ENT]description[SEP] labels and abstracts...", 1)
ordered_entity_embeddings = extend_entity_dataset(kge, entities_data, tokenizer)



# return ordered_mention_contexts, None


# for k in eg:
#     file_data = eg[k]
#     for k2 in file_data:
#         text_entities = file_data[k2]
#         urls_list += text_entities
#
# urls_set = list(dict.fromkeys(urls_list))

