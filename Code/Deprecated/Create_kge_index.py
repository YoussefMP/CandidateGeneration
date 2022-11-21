from elasticsearch import Elasticsearch, helpers
from threading import Thread
import numpy as np


ES_CLIENT = Elasticsearch(hosts="https://datasets.es.us-central1.gcp.cloud.es.io:9243",
                          basic_auth=('elastic', 'rAk1VSiYoyvDn1kZJGV2ik32'),
                          request_timeout=120)


def generate_index(index, es_data):
    # if ES_CLIENT.indices.exists(index=index):
    #     print("deleting the '{}' index.".format(index))
    #     res = ES_CLIENT.indices.delete(index=index)
    #     print("Response from server: {}".format(res))
    #
    # ES_CLIENT.indices.create(index=index)

    actions = [
        {"_index": index,
         "_source": {
             index.split('-')[0]: key,
             index.split('-')[1]: np.array(es_data[key])
         }}
        for key in es_data
    ]

    responses = list(helpers.parallel_bulk(ES_CLIENT, actions, thread_count=10))
    print(f"Finished indexing this chunck")
    return responses


def read_kg_embeddings_file(fid):
    kge = {}
    line_count = 0
    input_file = open(f"../Data/vectors_{fid}.txt", "r", encoding="utf8")

    for line in input_file:

        line_count += 1

        vals = line.split(" ")
        kge[vals[0]] = []

        for val in vals[1:]:
            try:
                kge[vals[0]].append(float(val))
            except ValueError:
                if not val == "\n":
                    print(f"Embedding of {vals[0]} contains error: '{val}'")

    input_file.close()

    print(f"Finished reading file vectors_{fid}")

    instance_resp = generate_index("entity-embeddings", kge)

    return kge


if __name__ == "__main__":

    for file_id in range(1, 19, 2):

        t1 = Thread(target=lambda: read_kg_embeddings_file(file_id))
        t2 = Thread(target=lambda: read_kg_embeddings_file(file_id+1))

        print(f"Starting thread 1 and 2")
        t1.start()
        t2.start()

        if file_id == 17:
            print(f"Starting thread 3 for this iteration")
            t3 = Thread(target=lambda: read_kg_embeddings_file(file_id+2))
            t3.start()
            t3.join()

        t1.join()
        t2.join()




