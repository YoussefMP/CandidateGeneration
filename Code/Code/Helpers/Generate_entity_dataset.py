import requests
import json
from Code.Logger import log

headers = {
    'Content-Type': 'application/json',
}


def get_index_entities(index, size=0):
    """
        Get {size} urls of entities present in the index
    :param index:   index name
    :param size:    nb of url requested
    :return: {"entitylist": [str: URL, ]}
    """

    json_data = {"indexname": index}
    if size != 0:
        json_data['size'] = size

    log("sent request to server and waiting on response ...", 1)
    response = requests.get('http://unikge.cs.upb.de:5001/get-all-entity', headers=headers, json=json_data)
    try:
        return json.loads(response.text)["entitylist"]
    except json.decoder.JSONDecodeError:
        print(response.text)


def get_entity_embedding(entities, index):
    """
        get the embedding of the entities in the list
    :param entities:  [str: URL, ...]
    :param index:     index name
    :return: dict:    {str:URL : [float, ] }
    """
    json_data = {
        'indexname': index,
        'entities': entities,
    }
    response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding', headers=headers, json=json_data)
    try:
        return json.loads(response.text)
    except json.decoder.JSONDecodeError:
        print(response.text)


def load_entity_embeddings_dict(index_name, size):
    """
    :param size:
    :param index_name:
    :return:  {str:URL: [float, ...]}
    """
    log("Requesting entities urls from the server", 1)
    log("...", 1, end=True)
    entities_list = get_index_entities(index_name, size)
    log(f"Got all the entities urls requested {size}", 1)

    embeddings_mappings = {}
    log("Requesting entities embeddings from the server ...", 1)
    for idx in range(10, len(entities_list), 10):

        if idx % 1000 == 0:
            log(f"we have processed {(idx/len(entities_list))*100}% of the queries...", 2)

        ans = get_entity_embedding(entities_list[idx-10: idx], index_name)
        for url, emb in ans.items():
            embeddings_mappings[url] = emb

        if idx + 10 > len(entities_list)-1:
            ans = get_entity_embedding(entities_list[idx:], index_name)
            for url, emb in ans.items():
                embeddings_mappings[url] = emb
    log("Finished retrieving the embeddings from the database", 1)
    return embeddings_mappings


def filter_entities_for_description(entities_embeddings, criteria=None):
    """
    Read the DBpedia dump file and extract labels and descriptions of the entities present in the dataset
    :param entities_embeddings:
    :param criteria
    :return: complete_entities_dataset {txt_id: [{link: str: DBpedia_link,
                                                 label: str: entity label,
                                                 abstract: str: entity description
                                                }]
                                        }
    """
    import re
    import torch

    file_path = f".\\..\\..\\Data\\DBpedia dumps\\"
    if criteria == "label":
        file_path += "labels_lang=en.ttl"
        regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<label>.*)"@en''')
    else:
        file_path += "short-abstracts_lang=en.ttl"
        regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<abstract>.*)"@en''')

    keys = list(entities_embeddings.keys())
    keys.sort()
    complete_entities_dataset = {}
    count = 0

    log(f"Filtering entities with no {criteria}", 1)

    with open(file_path, "r", encoding="utf-8") as file:
        line = "a"
        t = 0
        while line:
            t += 1
            line = file.readline()
            m = regex.match(line)

            if m:
                if m.group("link") in keys:
                    count += 1
                    try:
                        if criteria == "label":
                            complete_entities_dataset[m.group("link")] =\
                                (m.group("label"), torch.FloatTensor(entities_embeddings[m.group("link")]))
                        else:
                            complete_entities_dataset[m.group("link")] = m.group("abstract")
                    except KeyError as k_err:
                        print(f"KeyError: {k_err} \n ")
                    keys.remove(m.group("link"))

            if t % 1500000 == 0:
                log(f"Parsed = {t} lines, and found data for {count}/{len(entities_embeddings.keys())} entities"
                    f" ////  {len(keys)}", 3)

            if len(entities_embeddings.keys()) == count and criteria == "label":
                break
        log(f"Results of filtering entities with no {criteria} => {count} / {len(entities_embeddings)} = "
            f"{count/len(entities_embeddings)}%", 2)

    file.close()
    return complete_entities_dataset


def write_complete_entity_dataset(labels, abstracts):
    """
    Write tsv file: Row Format = URL \t Embedding \t Label \t Description
    :param labels:
    :param abstracts:
    :return:
    """
    import csv
    file_path = "./../../Data/Results/complete_entity_dataset.tsv"

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


def main(index_name, size):

    log("Building dict of entities embeddings ...", 0)
    entity_embeddings_dict = load_entity_embeddings_dict(index_name, size)
    log("Filtering for available data ...", 1)
    entity_label_dict = filter_entities_for_description(entity_embeddings_dict, criteria="label")
    entity_abstract_dict = filter_entities_for_description(entity_embeddings_dict, criteria="abstract")
    log("Writing results to file ...", 1)
    entities_data = write_complete_entity_dataset(entity_label_dict, entity_abstract_dict)


nb_of_urls = 3000

# main("shallom_dbpedia_index")
main("transe_dbpedia_l2_entity", nb_of_urls)
