import os
import sys

sys.path.insert(0, os.getcwd())

from Code.FileIOManager import ModelFileIO

fman = ModelFileIO()

kge_path = f"{fman.repo}/Datasets/TransE/entity_embeddings_dbp21-03_transe_dot.tsv"
miss_path = f"{fman.repo}/Datasets/TransE/missing.txt"
out_path = f"{fman.results_repo}/missing_emb.tsv"

f1 = open(kge_path, "r", encoding="utf-8")
f2 = open(miss_path, "r", encoding="utf-8")
f3 = open(out_path, "w", encoding="utf-8")

import csv

missing = f2.readlines()
for i in range(len(missing)):
  missing[i] = missing[i].strip("\n")

missing = list(dict.fromkeys(missing))

import time

start = time.time()
f1_lines = f1.readlines()
end = time.time()
print(f"to readlines I needed {end -start}")

print("cleaning lines...")
start = time.time()
lines = [line.split("\t")[0] for line in f1_lines]
print(f"lines[0] ===> {lines[0]}")
end = time.time()
print(f"{end-start} time for cleaning")

reader = csv.reader(f1, delimiter="\t")

writer = csv.writer(f3, delimiter="\t")

found = 0
entries = 0

print("Entering the for loop")
for url in missing:

  entries += 1
  if entries % 100000 == 0:
    print("Iterated {entries} lines from {len(lines)} and found {found}")


  if url in lines:
    found += 1
    writer.writerow(f1_lines[lines.index(url)].split("\t"))

    if found == 100:
      print("found {found} links")

f2.close()
f1.close()

import sys
sys.exit()


count = 0
found = 0
dlinks = 0

print(f"I have {len(missing)}")

#import pdb
#pdb.set_trace()

for row in reader:
  if count % 5000000 == 0:
    print(f"from {len(missing)} we found {found} after processing {count} lines which {dlinks} from them are dbpedia links")
  count += 1

#  if count < 76000000:
#   continue

  if "dbpedia" in row[0]:
    dlinks += 1
    if row[0] in missing:
      found += 1
      writer.writerow(row)
      missing.pop(missing.index(row[0]))

  if found == 2160:
    break
