import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import csv
import gc
from Code.FileIOManager import ModelFileIO, time_format
from Code.blink.Dataset_processing import str_to_tensor
import threading
from Code.blink.biencoder import BiEncoderRanker
from torch.utils.data import DataLoader

global input_tokens_1
global input_tokens_2
global url_map_1
global input_embeddings_1
global input_embeddings_2
global url_map_2

__DEBUG__ = True

class GraphData:
    def __init__(self, url_map, tokens_1, emb_1):
        self.x = []
        self.y = []
        self.z = []

        for tid in range(len(tokens_1)):
          self.z.append(url_map[tid])
          self.x.append(tokens_1[tid].clone().detach())
          self.y.append(emb_1[tid].clone().detach())

#        for tid in range(len(tokens_2)):
#          self.x.append(tokens_2[tid].clone().detach())
#          self.y.append(emb_2[tid].clone().detach())

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return self.n_samples


def read_file(t):

    if t == 1:
        print(f"Reading file ===> {fman.results_repo}")
        file = open(f"{fman.results_repo}/new_entries.tsv", "r", encoding="utf-8")
        reader = csv.reader(file, delimiter="\t")
        global input_tokens_1
        global input_embeddings_1
        global url_map_1

    else:
        print(f"Reading file ===> {fman.results_repo}")
        file = open(f"{fman.results_repo}/new_entries2.tsv", "r", encoding="utf-8")
        reader = csv.reader(file, delimiter="\t")
        global input_tokens_2
        global input_embeddings_2
        global url_map_2

    print(f"Thread_{t}: Reading file_rows")
    lines = 0
    for row in reader:

        token_ids = str_to_tensor(row[1])
        embedding = str_to_tensor(row[2])

        lines += 1

        token_ids = torch.IntTensor(token_ids)
        embedding = torch.FloatTensor(embedding)

        if lines == 1:
          print(f"token_ids ===> {token_ids.size()}")
          print(f"embeddings ===> {embedding.size()}")

        if t == 1:
            url_map_1.append(row[0])
            input_tokens_1.append(token_ids)
            input_embeddings_1.append(embedding)
        else:
            url_map_2.append(row[0])
            input_tokens_2.append(token_ids)
            input_embeddings_2.append(embedding)
#    print(f"Clean ={clean}  ///// dirty = {dirty}")
        if __DEBUG__ and lines > 100000:
          break

def load_model():

    import json
    config_name = "/biencoder_wiki_large.json"

    with open(f"{fman.models_repo}{config_name}", "r", encoding="utf-8") as j_file:
        params = json.load(j_file)
        params["kge_emb_dim"] = 100
    j_file.close()

    reranker = BiEncoderRanker(params)
    print("Loading the model's weights...")
    reranker.load_model(f"{fman.models_repo}/BLINK_output/pytorch_model.bin")

    return reranker.to(device)


def write_new_emb(t):
    global input_tokens_1
    global input_embeddings_1
    global url_map_1

    global input_tokens_2
    global input_embeddings_2
    global url_map_2

    if t == 1:
      print("preparing the Dataset")
      dataset = GraphData(url_map_1[:len(url_map_1)//2], input_tokens_1[:len(input_tokens_1)//2], input_embeddings_1[:len(input_embeddings_1)//2])
      dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4 )

      out_file = open(f"{fman.results_repo}/new_kge_embeddings_11.tsv", "w", encoding="utf-8")
      writer = csv.writer(out_file, delimiter="\t")

    elif t == 0:
      print("preparing the Dataset")
      dataset = GraphData(url_map_1, input_embeddings_1)
      dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4 )

      out_file = open(f"{fman.results_repo}/new_kge_embeddings_11.tsv", "w", encoding="utf-8")
      writer = csv.writer(out_file, delimiter="\t")


    elif t == 2:
      print("preparing the Dataset")
      dataset = GraphData(url_map_1[len(url_map_1)//2:], input_tokens_1[len(input_tokens_1)//2:], input_embeddings_1[len(input_embeddings_1)//2:])
      dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)

      out_file = open(f"{fman.results_repo}/new_kge_embeddings_12.tsv", "w", encoding="utf-8")
      writer = csv.writer(out_file, delimiter="\t")

    elif t == 3:
      print("preparing the Dataset")
      dataset = GraphData(url_map_2[:len(url_map_2)//2], input_tokens_2[:len(input_tokens_2)//2], input_embeddings_2[:len(input_embeddings_2)//2])
      dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)

      out_file = open(f"{fman.results_repo}/new_kge_embeddings_21.tsv", "w", encoding="utf-8")
      writer = csv.writer(out_file, delimiter="\t")

    elif t == 4:
      print("preparing the Dataset")
      dataset = GraphData(url_map_2[len(url_map_2)//2:], input_tokens_2[len(input_tokens_2)//2:], input_embeddings_2[len(input_embeddings_2)//2:])
      dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)

      out_file = open(f"{fman.results_repo}/new_kge_embeddings_22.tsv", "w", encoding="utf-8")
      writer = csv.writer(out_file, delimiter="\t")

    print("Getting the first_batch...")
    for step, batch in enumerate(dataloader):

        if step == 0:
            start = time.time()
            print(f"\t\tbatch[0] = {batch[0][0].size()}")
            print(f"\t\tbatch[1] ==> {batch[0][1].size()}")
            print(f"\t\tbatch[2] ==> {batch[1].size()}")

        input_ids = batch[0].to(device)
        input_emb = batch[1].to(device)

        out = biencoder.encode_candidate(input_ids, input_emb)
        out.detach()
        input_ids.detach()
        input_emb.detach()
#        import pdb
#        pdb.set_trace()

        for idx in range(8):
            writer.writerow([batch[2][idx].tolist(), out[idx].tolist()])

        if step % 10 ==0:
            end = time.time()
            print(f"Finished batch {step} in {time_format(int(end-start))}")
            start = time.time()



if __name__ == "__main__":
    fman = ModelFileIO()

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(f"device ===> {device}")

    input_tokens_1 = []
    input_tokens_2 = []
    url_map_1 = []
    input_embeddings_2 = []
    input_embeddings_1 = []
    url_map_2 = []

    import time

    start = time.time()
    reader_1 = threading.Thread(target=lambda: read_file(1))
    reader_2 = threading.Thread(target=lambda: read_file(2))

    reader_1.start()
    reader_2.start()

    reader_1.join()
    print("reader one finished")
    reader_2.join()
    print("reader 2 finished")
    end = time.time()

    print(f"it took {end-start} time to finish reading the files ")

    # Load model
    biencoder = load_model()
    write_new_emb(0)
#    writer_1 = threading.Thread(target=lambda: write_new_emb(1))
#    writer_2 = threading.Thread(target=lambda: write_new_emb(2))
#    writer_3 = threading.Thread(target=lambda: write_new_emb(3))
#    writer_4 = threading.Thread(target=lambda: write_new_emb(4))

#    writer_1.start()
#    writer_2.start()
#    writer_3.start()
#    writer_4.start()
