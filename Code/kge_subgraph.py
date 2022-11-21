import csv
import sys
import os
sys.path.insert(0, os.getcwd())

from Code.FileIOManager import ModelFileIO, time_format
from Code.blink.Gen_subgraph_new_emb import get_aida_files_urls
import time
import random
import csv

fman = ModelFileIO()

ds_urls = get_aida_files_urls(fman)
ds_urls = list(dict.fromkeys(ds_urls))
#kge_file = open(f"{fman.results_repo}/sub_kge.tsv", "r", encoding="utf-8")
kge_file = open(f"{fman.repo}/Datasets/TransE/entity_embeddings_dbp21-03_transe_dot.tsv", "r", encoding="utf-8")
reader = csv.reader(kge_file, delimiter="\t")
sub_kge = {}

output_file = open(f"{fman.results_repo}/sub_kge_18_11_22.tsv", "w", encoding="utf-8")
writer = csv.writer(output_file, delimiter="\t")

found = 0
lid = 0
start = time.time()
for row in reader:

    if len(row) < 4:
      continue

    lid += 1
    url = row[0]

    if "dbpedia" not in url:
      continue

    if lid % 1000000 == 0:
        end = time.time()
        print(f"To process {lid} lines it took {time_format(int(end-start))} , and we found {found} links")
        start = time.time()

    if url in ds_urls:
        found += 1
        writer.writerow(row)
        continue

    u = random.uniform(0 ,1)
    if u < 0.3 and found < 2328500:
        found += 1
        writer.writerow(row)
        try:
          ds_urls.remove(url)
        except ValueError:
          pass

    if len(ds_urls) == 0 and found > 2328500:
        break

print(len(ds_urls))
