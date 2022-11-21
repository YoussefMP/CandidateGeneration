import elastic_transport
from elasticsearch import Elasticsearch, helpers
from threading import Thread
from Logger import log
import numpy as np
import requests
import csv


def fill_index(index, es_data, first=True):
    if first:
        if ES_CLIENT.indices.exists(index=index):
            log("deleting the '{}' index.".format(index), 2)
            res = ES_CLIENT.indices.delete(index=index)
            log("Response from server: {}".format(res), 2)

        ES_CLIENT.indices.create(index=index)
        first = False

    log(f"Called to index {len(list(es_data.keys()))} items...", 2)

    actions = [
        {"_index": index,
         "_source": {
             index.split('_')[0]: key,
             index.split('_')[1]: np.array(es_data[key])
         }}
        for key in es_data
    ]

    try:
        responses = list(helpers.parallel_bulk(ES_CLIENT, actions, thread_count=10))
    except elastic_transport.ConnectionTimeout:
        responses = None

    log(f"Finished indexing this chunk", 2)

    return responses


def read_kg_embeddings_file(f_path, fid):
    kge = {}
    line_count = 0
    input_file = open(f"{f_path}/vectors_{fid}.txt", "r", encoding="utf8")

    for line in input_file:

        if "dbpedia.org" not in line:
            continue

        line_count += 1

        vals = line.split(" ")
        kge[vals[0]] = []

        for val in vals[1:]:
            try:
                kge[vals[0]].append(float(val))
            except ValueError:
                if not val == "\n":
                    log(f"Embedding of {vals[0]} contains error: '{val}'", 1)

    input_file.close()

    log(f"Finished reading file vectors_{fid}", 1)

    if fid == 0:
        instance_resp = fill_index("entity_embeddings", kge, True)
    else:
        instance_resp = fill_index("entity_embeddings", kge, False)

    return kge


def read_csv_embeddings_file(file):

    log(f"Reading csv_file {file}", 0)
    data = open(file, 'r')
    reader = csv.reader(data, delimiter=',')

    es_data_dict = {}

    for row in reader:
        embedding = []
        if len(row) > 0:
            emb = row[1].strip('[').strip(']').split(' ')
            for val in emb:
                try:
                    embedding.append(float(val))
                except ValueError:
                    pass
            es_data_dict[row[0]] = embedding

    data.close()
    return es_data_dict


def read_csv_mapping_file(file):
    log(f"Reading csv_file {file}", 0)
    data = open(file, 'r', encoding='utf8')
    reader = csv.reader(data, delimiter=',')

    es_data_dict = {}

    for row in reader:
        if len(row) > 1:
            m_list = []
            for e in row[1:]:
                if e != " ":
                    m_list.append(e.strip(' '))

            es_data_dict[row[0]] = m_list

    data.close()
    return es_data_dict


if __name__ == "__main__":
    # Data path
    ds_path = "../Data"

