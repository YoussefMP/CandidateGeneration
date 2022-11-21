from Attention_Encoder import TransformerEncoder
from Data_Manager import EmbeddingsMyDataset
from FileIOManager import ModelFileIO
from torch.utils.data import DataLoader
from Logger import log
import torch.nn as nn
import requests
import torch
import json
import numpy as np

__DEBUG__ = False


def tensor_to_str(tensor):
    req_str = "["

    for val in tensor:
        req_str += str(val.item()) + ","

    req_str = req_str[:-1] + "]"
    return req_str


def get_k_neighbours(index_name, emb_batch, k=0):
    headers = {
        'Content-Type': 'application/json',
    }

    neighbours_list = []

    for embedding in emb_batch:

        embedding = tensor_to_str(embedding)
        data = f'''{{
            \"indexname\":\"{index_name}\",
            \"embedding\":{embedding},
            \"distmetric\":\"cosine\"
        }}'''

        response = requests.get('http://unikge.cs.upb.de:5001/get-embedding-neighbour',
                                headers=headers,
                                data=data
                                )

        try:
            json_obj = json.loads(response.text)
            neighbours_list.append(json_obj)

        except json.decoder.JSONDecodeError:
            print(f"Error decoding server's response =\n {response.text}")

    return neighbours_list


def eval_model(encoder, dataloader, index_name, set_size):
    # Training Hyper parameters
    log("Setting Hyperparameters for the model", 1)

    # Loss function and optimizer
    criterion = nn.CosineSimilarity()

    # Training Loop
    encoder.eval()

    tp_count = 0
    total_count = 0

    for idx, data in enumerate(dataloader):

        txt_id = data[0]
        mention = data[0][1]
        target_entities = data[1][0]
        source = data[0][2]
        target = data[1][1]

        with torch.no_grad():

            entry_size = int(768 / encoder.source_size)
            source = source.reshape([source.size(0), entry_size, -1])
            source = source.mean(1)

            # outputs = encoder(source, target)
            outputs = encoder(source)[1]            # New model evaluation input

        neighbours_list_batch = get_k_neighbours(index_name, outputs, set_size)

        cos_sims_batch_avg = []
        tid = 0
        for neighbours_list in neighbours_list_batch:
            sum_cos_sim = 0
            total_count += 1
            found = False
            for neighbour in neighbours_list["neighbours"]:

                sum_cos_sim += criterion(target[tid].unsqueeze(0), torch.FloatTensor(neighbour["embeddings"]).unsqueeze(0))
                if neighbour["entity"] in target_entities[tid]:
                    print(f"target_entity ==> {target_entities[tid]} / mention ==> {mention[tid]}")
                    # print(f"===> {neighbour['entity']}")
                    found = True
                    tp_count += 1
                    print(f"Predicted vector -> {outputs[tid]}")
                    print(f"Avg = {sum(outputs[tid])/len(outputs[tid])}, F/L = {outputs[tid][0]}, {outputs[tid][-1]}")
                    print(f"____________ Found = {found} {tp_count}/{total_count} ______________")

                print("ran Batch with 0 Results")
            tid += 1
            cos_sims_batch_avg.append(sum_cos_sim / 10)

    batch_cos_sim_avg = sum(cos_sims_batch_avg) / len(cos_sims_batch_avg)
    gold_recall = (tp_count * 100) / total_count
    return {"gold_recall": gold_recall, "batch_cos_sim_avg": batch_cos_sim_avg}


def eval_biencoder_model(biencoder, dataloader, index_name, ):
    # Training Hyper parameters
    log("Setting Hyperparameters for the model", 1)

    # Loss function and optimizer
    criterion = nn.CosineSimilarity()

    # Training Loop
    biencoder.bi_encoder.eval()

    tp_count = 0
    total_count = 0

    for idx, data in enumerate(dataloader):

        source = data[0]
        target_entity_batch = data[1][0]
        target_embeddings_batch = data[1][1]

        with torch.no_grad():
            from blink.blink_Core import to_bert_input

            source, attention_mask, segments_idx = to_bert_input(source, 0)
            ctxt_outputs = biencoder.bi_encoder(source, attention_mask, segments_idx, None, None, None)[0]

        # Number of candidates to be returned from the index for each mention

        # for output in ctxt_outputs:
        k = 10
        # D, candidates = biencoder.faiss_index.search(output, k)
        D, candidates_batch = biencoder.faiss_index.search(np.array(ctxt_outputs), k)

        cos_sims_batch_avg = []

        for candidates in candidates_batch:

            sum_cos_sim = 0
            total_count += 1
            found = False
            tid = 0

            for candidate in candidates:
                candidate_entiy = biencoder.hot_index[candidate][0]
                candidate_embeddings = biencoder.hot_index[candidate][1]

                sum_cos_sim += criterion(target_embeddings_batch[tid].unsqueeze(0), candidate_embeddings)

                if candidate_entiy in target_entity_batch[tid]:
                    tp_count += 1
                    print(f"found == > {target_entity_batch[tid]} under {[biencoder.hot_index[x][0] for x in candidates]}")
                    print(f"____________ Found = {found} {tp_count}/{total_count} ______________")

            tid += 1
            cos_sims_batch_avg.append(sum_cos_sim / 10)

    batch_cos_sim_avg = sum(cos_sims_batch_avg) / len(cos_sims_batch_avg)
    gold_recall = (tp_count * 100) / total_count
    print(f"____________ Found = {found} {tp_count}/{total_count} ______________")
    return {"gold_recall": gold_recall, "batch_cos_sim_avg": batch_cos_sim_avg}
