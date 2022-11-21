import requests
import json
import csv
import torch
from Logger import log


def get_entity_embedding(url, index_name):
    headers = {
        'Content-Type': 'application/json',
    }
    if "shallom" in index_name:
        url = url.replace("http://dbpedia.org", "")

    json_data = {
        'indexname': index_name,
        'entities': [
            url,
        ],
    }

    response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding', headers=headers, json=json_data)

    try:
        ans = json.loads(response.text)
        tensor = torch.FloatTensor(ans[url]).unsqueeze(0)
        return tensor
    except json.decoder.JSONDecodeError:
        return None


def gen_ent_emb_index_from_file(path, index_name):
    data = open(path, 'r', encoding='utf8')
    reader = csv.reader(data, delimiter='\t')

    t_count = 0
    p_count = 0

    entity_emb = {}

    for row in reader:

        if row[1] is entity_emb.keys():
            print("found a double")
            continue

        res = get_entity_embedding(row[1], index_name)
        t_count += 1

        if res is None:
            continue
        else:
            p_count += 1
            entity_emb[row[1]] = res

        if (t_count % 200) == 0:
            log(f"processed {t_count} urls and found {p_count} embveddings", 1)

    return entity_emb, t_count, p_count


def gen_ent_emb_index_from_list(eg_list, index_name):
    t_count = 0
    p_count = 0

    entity_emb = {}

    for k, v_list in eg_list.items():
        for ent in v_list:
            if ent in entity_emb.keys():
                continue

            res = get_entity_embedding(ent, index_name)
            t_count += 1

            if res is None:
                continue
            else:
                p_count += 1
                entity_emb[ent] = res

            if (t_count % 200) == 0:
                log(f"processed {t_count} urls and found {p_count} embveddings", 1)

    return entity_emb


def write_results_file(index, path):
    file = open(path, "w", encoding='utf8')
    writer = csv.writer(file, delimiter="\t")

    for url, embedding in index.items():
        writer.writerow([url, embedding])

    file.close()


if __name__ == "__main__":
    indx_name = "shallom_dbpedia_index"
    ent_emb_index, t, p = gen_ent_emb_index_from_file("./../../Data/Results/processed_aida_complete.tsv", indx_name)

    print(f"out of {t}-url we found {p} embeddings")

    write_results_file(ent_emb_index)

    # Testing file readability
    # file = open("./../../Data/Results/entity_embeddings(2).tsv", "r", encoding='utf8')
    # reader = csv.reader(file, delimiter="\t")
    #
    # c = 0
    # for row in reader:
    #     if row:
    #         print(f"row[0] = {row[0]}\nrow[1] = {row[1]}")
    #         c += 1
    #     if c == 10:
    #         break
