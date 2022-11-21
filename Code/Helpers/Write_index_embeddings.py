import requests
import json
import csv
import time


def time_format(seconds: int):
    if seconds is not None:
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        if d > 0:
            return '{:02d}D {:02d}H {:02d}m {:02d}s'.format(d, h, m, s)
        elif h > 0:
            return '{:02d}H {:02d}m {:02d}s'.format(h, m, s)
        elif m > 0:
            return '{:02d}m {:02d}s'.format(m, s)
        elif s > 0:
            return '{:02d}s'.format(s)
    return '-'



headers = {
    'Content-Type': 'application/json',
}

json_data = {
    'indexname': 'transe_dbpedia_l2_entity',
}

print(f"Querying the server for the entities...")

start = time.time()
response = requests.get('http://unikge.cs.upb.de:5001/get-all-entity', headers=headers, json=json_data)
end = time.time()

print(f"processing the request took about {time_format(end-start)}")

j_obj = json.loads(response.text)

# TODO: save the entities in a list
entities_list = j_obj["entitylist"]

print(f"Got the entities & stored in list of length {len(entities_list)}")
embeddings = {}

for i in range(20, len(entities_list), 20):
    json_data = {
        'indexname': 'transe_dbpedia_l2_entity',
        'entities': entities_list[i-20: i],
    }
    response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding', headers=headers, json=json_data)
    answer = json.loads(response.text)
    embeddings = {**embeddings, **answer}

    if i+20 > len(entities_list):
        json_data = {
            'indexname': 'transe_dbpedia_l2_entity',
            'entities': entities_list[i: ],
        }
        response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding', headers=headers, json=json_data)
        answer = json.loads(response.text)
        embeddings = {**embeddings, **answer}

    if i % 500000 == 0:
        print(f"{i} entities have been retrieved from the index")


with open(f"./../../Data/Results/complete_transe_embeddings.tsv", "w", encoding="utf-8") as out_file:
    writer = csv.writer(out_file, delimiter="\t")
    for k, v in embeddings.items():
        writer.writerow([k, v])
