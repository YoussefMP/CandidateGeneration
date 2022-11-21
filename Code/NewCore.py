from torch.utils.data import DataLoader
import argparse
from Logger import log
from Code.blink.indexer.faiss_indexer import DenseHNSWFlatIndexer
from FileIOManager import ModelFileIO, time_format, str_to_tensor
from Code.Attention_Encoder import initiate_model
from Code.Core import input_loop, MODEL_PARAMS
from Code.Data_Manager import FinalDataset
from MentionsEncoder import Encoder
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import numpy as np
import faiss
import torch
import time
import csv
import os
import random


__DEBUG__ = False
NEGATIVES = "HARD"
#LOSS_FUNC = "BCELossWithLogits"
# LOSS_FUNC = "COSSME"
LOSS_FUNC = "BCECosSim"
NEG_SAMPLES = 7
FILE_NAME = {"context": "context_Final_pair_files.tsv",
             "nocontext": "Final_pair_files.tsv",
             "4context": "4layers_context.tsv",
             "4nocontext": "4layers_nocontext.tsv"
             }
method = "4context"
TOP = 0
clipping_norm = 10
url_map_name = "small_url_map4.txt"
index_name = "test_index4"


def read_final_file(fman):
    if TOP != 0:
        entries = fman.load_specific_data(f"{fman.results_repo}/{FILE_NAME[method]}", TOP)
    else:
        file = open(f"{fman.results_repo}/{FILE_NAME[method]}", "r", encoding="utf=8")
        reader = csv.reader(file, delimiter="\t")

        entries = []
        for row in reader:
            if len(row) > 2:
                entries.append([row[0], row[1], row[2], str_to_tensor(row[3])])

                if len(entries) % 3000 == 0:
                    log(f"processed {len(entries)} rows", 1)
                    if __DEBUG__:
                        break
            # if len(entries) == 1000:
            #     break

    return entries, "From File"


def read_dataset(fman):
    """
    Read the data from the aida files
    :param fman: file_manager object
    :return: d={tid: (url, mention, embedding)}, tc={text_id:tuple(int): text:str}
    """

    if os.path.exists(f"{fman.results_repo}/{FILE_NAME[method]}"):
        return read_final_file(fman)

    data_files = ["aida_train", "aida_testa", "aida_testb"]
    file_types = ["training", "validation", "evaluation"]

    for i in range(3):
        fman.add_path(file_types[i], data_files[i], extension="Datasets")

    texts, eg_dict, mentions_dict, smc = fman.read_aida_input_doc("training")

    encoder = Encoder()
    # Get the datamap {text_id, url, mention, mention_embeddings}
    # d, tc = encoder.embed_mentions({i: texts[i] for i in range(3)}, eg_dict, mentions_dict)
    with_context = True if "context" in method else False
    d, tc = encoder.embed_mentions(texts, eg_dict, mentions_dict, with_context)

    fo = open(f"{fman.results_repo}/{FILE_NAME[method]}", "w", encoding="utf-8")
    writer = csv.writer(fo, delimiter="\t")

    for val in d:
        writer.writerow([val[0], val[1], val[2], val[3]])

    fo.close()
    return d, tc


def load_index_from_file(fman):

    url_map = []
    d = 100  # Dimensionality of our vectors
    index = DenseHNSWFlatIndexer(d, store_n=64, ef_search=64, ef_construction=64)

    s = time.time()
    index.index = faiss.read_index(f"{fman.results_repo}/{index_name}")
    e = time.time()
    log(f"{time_format(e-s)} for loading the index from file", 1)

    url_f = open(f"{fman.results_repo}/{url_map_name}", "r", encoding="utf-8")
    line = "a"
    k = 0
    s = time.time()
    while line:
        line = url_f.readline()
        url_map.append(line.strip("\n"))
    e = time.time()
    log(f"{e-s} is the time for reading the urls file", 2)
    return index, url_map


def load_transe_index(fman, links=None):
    """
    loads the TransE embeddings into a FAISS index
    :return:
    """
    if os.path.exists(f"{fman.results_repo}/{url_map_name}"):
        return load_index_from_file(fman)

    fman.add_path("index", "sub_sub_kge.tsv", "Datasets")

    log("Opening kge file...", 1)
    file = open(f"{fman.files['index']}", "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

    entries = 0
    url_map = []
    index_vector = []

    start = time.time()
    for row in reader:

        if len(row) > 100:
            if random.uniform(0, 1) < 0.25 or row[0] in links:
                entries += 1
                url_map.append(row[0])
                temp_vec = []
            else:
                continue

            for e in row[1:]:
                if len(temp_vec) < 100:
                    e = e.strip("\"\n")
                    temp_vec.append(float(e))
        else:
            continue

        if entries % 500000 == 0:
            end = time.time()
            log(f"Processed {entries} rows in {time_format(int(end - start))}", 2)
            start = time.time()

        index_vector.append(temp_vec)

    index_start = time.time()
    d = 100  # Dimensionality of our vectors
    index = DenseHNSWFlatIndexer(d, store_n=64, ef_search=32, ef_construction=32)

    log("Adding data to index...", 1)
    data_vec = np.array(index_vector, dtype=np.float32)
    index.index_data(data_vec)
    index_end = time.time()

    log(f"It took {time_format(int(index_end-index_start))} to finish building the index...", 0)

    faiss.write_index(index.index, f"{fman.results_repo}/{index_name}")

    url_io = open(f"{fman.results_repo}/{url_map_name}", "w", encoding="utf-8")
    for u in url_map:
        url_io.write(u + "\n")

    return index, url_map


def prepare_training_data(dataset, kge, url_list, batch_size=0):

    log(f"Preparing the data for training...", 1)
    start = time.time()
    mapped_data = []

    entered_pairs = 0
    process = 0
    cache_hit = 0

    cached_entries = {}

    log(f"{len(dataset)} mentions to find, We have {len(url_list)} links in the KGE", 2)

    for entry in dataset:
        process += 1

        if process % 3000 == 0 or process == len(dataset):

            log(f"From {len(dataset)} processed {process} and found {entered_pairs} URLS in the KGE from which "
                f"{cache_hit} are cache hits", 2)

        url = entry[1]
        m_emb = entry[3]

        if url not in cached_entries.keys():
            try:
                eid = url_list.index(url)
                entered_pairs += 1
            except ValueError:
                continue

            target_vec = kge.index.reconstruct(eid)[:-1]
            cached_entries[url] = target_vec

        else:
            target_vec = cached_entries[url]
            cache_hit += 1

        mapped_data.append([url, m_emb, target_vec])

        if entered_pairs == len(dataset):
            break

    dataset = FinalDataset(mapped_data)
    bs = 16 if batch_size == 0 else batch_size
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    end = time.time()

    log(f"Time needed to load all data into memory ==> {time_format(int(end-start))}\n")
    log(f"From {len(dataset)} we extracted {entered_pairs + cache_hit} pairs that will be used for training")

    return dataloader


def call_model(pretrained=False):

    def ask_for_params(parms):
        results = {}
        for key in parms.keys():
            results[key] = input_loop(key)
        return results

    if not os.path.exists(f"./../Models/"):
        os.makedirs("./../Models/")
    # d_res = ask_for_params(MODEL_PARAMS)
    # source_size, target_size, nb_layers, nb_heads, fwd_expansion, dp = d_res.values()

    # model_params = {"source_size": source_size if source_size else MODEL_PARAMS["source_size"],
    #                 "target_size": target_size if target_size else MODEL_PARAMS["target_size"],
    #                 "nb_layers": nb_layers if nb_layers else MODEL_PARAMS["nb_layers"],
    #                 "nb_heads": nb_heads if nb_heads else MODEL_PARAMS["nb_heads"],
    #                 "fwd_expansion": fwd_expansion if fwd_expansion else MODEL_PARAMS["fwd_expansion"],
    #                 "dp": dp if dp else MODEL_PARAMS["dp"]}

    model_params = {"source_size": 768,
                    "target_size": 100,
                    "nb_layers": 12,
                    "nb_heads": 12,
                    "fwd_expansion": 4,
                    "dp": 0.1}

    print(f"instantiating model with {model_params}")

    if not pretrained:
        models_path = f"{file_manager.models_repo}/{model_name}.bin"
        encoder = initiate_model(models_path, **model_params)
    else:
        return None, model_params

    return encoder, model_params


def get_scores_and_labels(fidx, url_list, qv, target, eg, neg_type):

    neg_samples = NEG_SAMPLES

    if neg_type == "HARD":
        D, I = fidx.search_knn(target.detach().numpy(), neg_samples)
    else:
        D, I = fidx.search_knn(qv.detach().numpy(), neg_samples)

    candidates_batch = []
    batch_labels = []

    for cids in range(len(I)):

        added_eg = False
        candidates = []
        labels = []

        for cid in I[cids]:
            candidates.append(fidx.index.reconstruct(int(cid))[:-1])
            if url_list[cid] == eg[cids]:
                labels.append(1)
            else:
                labels.append(0)

            u = random.uniform(0, 1)
            if u < 0.5 and len(labels) < 2*NEG_SAMPLES:
                candidates.append(target[cids].detach().numpy())
                labels.append(1)

        while len(labels) < 2*NEG_SAMPLES:
            candidates.append(target[cids].detach().numpy())
            labels.append(1)

        batch_labels.append(labels)
        candidates_batch.append(np.array(candidates))             # [32, 11, 100]

    try:
        candidates_batch = torch.FloatTensor(np.array(candidates_batch))
    except TypeError:
        for i in candidates_batch:
            print(i.shape)
    scores = torch.bmm(qv.unsqueeze(1), candidates_batch.permute(0, 2, 1))

    return scores, torch.FloatTensor(batch_labels), candidates_batch


def compute_my_loss(fidx, url_list, output, target, eg):

    cr = nn.CosineSimilarity()
    loss_batch = []

    neg_samples = NEG_SAMPLES
    D, I = fidx.search_knn(target.detach().numpy(), neg_samples)

    batch_loss = 0
    batch_candidates = []
    batch_labels = []

    for pid in range(output.size(0)):

        candidates = []
        labels = []

        for cid in I[pid]:
            candidates.append(fidx.index.reconstruct(int(cid))[:-1])
            if url_list[cid] == eg[pid]:
                labels.append(1)
            else:
                labels.append(0)

            u = random.uniform(0, 1)
            if u < 0.5 and len(labels) < 2*NEG_SAMPLES:
                candidates.append(target[pid].detach().numpy())
                labels.append(1)
        while len(labels) < 2*NEG_SAMPLES:
            candidates.append(target[pid].detach().numpy())
            labels.append(1)
        batch_candidates.append(candidates)
        batch_labels.append(labels)

    prob = cr(output.unsqueeze(2), torch.FloatTensor(np.array(batch_candidates)).permute(0, 2, 1))
    loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
    loss = loss_fct(prob, torch.FloatTensor(batch_labels))

    return loss


def train_model(fman, translator, dl, epochs, learning_rate, fidx, url_list):
    if not os.path.exists(f"{fman.models_repo}/{model_name}/"):
        os.makedirs(f"{fman.models_repo}/{model_name}/")
    step = 0
    log("Entering Epoch loop", 1)

    cr = nn.CosineSimilarity()
    cr2 = nn.MSELoss()
    criterion = nn.BCELoss()
    min_loss = -np.inf
    backed_losses = 0

    # set model in training mode
    translator.train()

    # Training Loop
    for epoch in range(epochs):
        e_start = time.time()
        log(f"____ Going through Epoch {epoch} ________ ", 1)

        optimizer = optim.Adam(translator.parameters(), lr=learning_rate)

        losses = []
        cos_sims = []

        for idx, data in enumerate(dl):
            eg = data[0]
            source = data[1]
            target = data[2]

            entry_size = int(768 / translator.source_size)
            source = source.reshape([source.size(0), entry_size, -1])
            source = source.mean(1)

            _, output = translator(source)

            if LOSS_FUNC != "BCECosSim":
                scores, labels, cbatch = get_scores_and_labels(fidx, url_list, output, target, eg, NEGATIVES)

            if LOSS_FUNC == "BCELossWithLogits":
                if idx == 0:
                    log(f"Loss function => {LOSS_FUNC}", 2)
                loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
                loss = loss_fct(scores.squeeze(), labels)

            elif LOSS_FUNC == "BCECosSim":
                if idx == 0:
                    log(f"Loss function => {LOSS_FUNC}", 2)
                loss = compute_my_loss(fidx, url_list, output, target, eg)

            else:
                if idx == 0:
                    log(f"Loss function => {LOSS_FUNC}", 2)
                loss = 1-cr(output, target) + cr2(output, target)
                loss = loss.mean()

            losses.append(loss.item())

            ########################################################################
            # if len(losses) == 1:
            #     min_loss = losses[-1]
            #     backed_losses += 1
            #     loss.backward()
            #
            # # elif loss.item() < min_loss or backed_losses = 0:
            # elif loss.item() <= losses[-1] and backed_losses < 5:
            #     backed_losses += 1
            #     loss.backward()
            #     if loss.item() < min_loss:
            #         min_loss = loss.item()
            #
            # elif loss.item() > min_loss and backed_losses < 5:
            #     log(f"Optimizing after accumulating {backed_losses} gradients...")
            #     if backed_losses == 0:
            #         loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     backed_losses = 0
            #
            # elif backed_losses == 5:
            #     log(f"Stepping after 5 backed losses {backed_losses} gradients...")
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     backed_losses = 0
            #
            # else:
            #     log("Skipped -- ", 0)
            ########################################################################

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if clipping_norm != 0:
                torch.nn.utils.clip_grad_norm_(translator.parameters(), max_norm=clipping_norm)

            if (idx % 10) == 0 and idx > 1 and LOSS_FUNC == "BCELossWithLogits":
                cos_scores = cr(output.unsqueeze(2), cbatch.permute(0, 2, 1))
                eg_scores = []
                for lid in range(labels.size(0)):
                    eg_id = labels[lid].tolist().index(1)
                    eg_scores.append(cos_scores[lid][eg_id].item())

                cos_sims.append(sum(eg_scores) / len(eg_scores))

                step += 1
                e_end = time.time()
                log(f"Epoch_{epoch}: is {(idx * 100) / dl.__len__()}% Done,"
                    f"Achieved loss {losses[-1]} at step:{step} / "
                    f"\n\t\t\t Highest scoring pred = {max(eg_scores)} ====> {eg[eg_scores.index(max(eg_scores))]}"
                    f"\n\t\t\t Average Golden score ===> {sum(eg_scores) / len(eg_scores)}"
                    f"\n\t\t\t time for batch process: { time_format(int(e_end - e_start))}", 2)

            elif (idx % 10) == 0 and idx > 1:
                e_end = time.time()
                cos_scores = cr(output, target)
                log(f"Epoch_{epoch}: is {(idx * 100) / dl.__len__()}% Done, loss of last batch = {loss}"
                    f"\n\t\t\t Highest scoring pred = {max(cos_scores)} ==="
                    f"=> {eg[torch.argmax(cos_scores)]}"
                    f"\n\t\t\t Average batch cos_sim = {cos_scores.mean()}"
                    f"\n\t\t\t time for batch process: {time_format(int(e_end - e_start))}"
                    )

                cos_sims.append(cr(output, target).mean().item())

        figure, axis = plt.subplots(2)
        axis[0].plot(losses)
        axis[0].set_title("BCEWithLogits Loss")

        axis[1].plot(cos_sims)
        axis[1].set_title("Average cosine similarity pro step")

        plt.savefig(f"{fman.models_repo}/{model_name}/Figure_epoch_{epoch}")

        e_end = time.time()
        log(f"Epoch_{epoch}: Avg-Loss = {sum(losses) / len(losses)}, Epoch lasted: {e_end - e_start}", 0)
        # log(f"Epoch_{epoch}: Avg-Loss = {sum(losses) / len(losses)}, Epoch lasted: {e_end - e_start}", 0)
        # time_sum += e_end - e_start

    torch.save(translator.state_dict(), f"{fman.models_repo}/{model_name}/{model_name}.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelname",
                        help="the name of the file in which the model will be saved after training", required=True)
    # parser.add_argument("--datafolder",
    #                     help="path to the folder containing the data files", required=True)
    # Training's hyperparameter
    parser.add_argument("--batchsize", required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--lr", help="learning rate", required=True)
    # Candidates set's size
    parser.add_argument("--setsize", help="Size of the set of candidates to be generated", required=True)
    args = parser.parse_args()

    model_name = args.modelname

    training_params = {
        "batch_size": int(args.batchsize),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "loss_function": LOSS_FUNC
    }

    file_manager = ModelFileIO()

    log(f"Reading the aida datasets from the files...")
    data, text_chunks = read_dataset(file_manager)

    # TODO Save the url in the dataset
    # If in dataset and 50% add

    links = []
    for d in data:
        if d[1] not in links:
            links.append(d[1])

    log("Loading the TranE embeddings")
    f_index, urls = load_transe_index(file_manager, links=links)

    if __DEBUG__:
        dataloader = prepare_training_data(data[:50], f_index, urls, batch_size=training_params["batch_size"])
    else:
        dataloader = prepare_training_data(data, f_index, urls, batch_size=training_params["batch_size"])

    model, params = call_model()

    # from Code.QuickModel import QModel
    # model = QModel(768, 100)

    train_model(file_manager, model, dataloader, training_params["epochs"], training_params["learning_rate"], f_index, urls)

    config_file = open(f"{file_manager.models_repo}/{model_name}/config_file.txt", "w", encoding="utf-8")
    for k in training_params:
        config_file.write(f"{k}={training_params[k]}\n")
    for k in params:
        config_file.write(f"{k}={params[k]}\n")

    config_file.write(f"clip_norm={clipping_norm}")
    config_file.write(f"Loss-function={LOSS_FUNC}")

# TODO: add mention to the batch (preparing training data)
