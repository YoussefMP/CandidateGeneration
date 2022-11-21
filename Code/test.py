import requests
import json
import torch

headers = {
    'Content-Type': 'application/json',
}


# ____________________________________________ Get list of indices
print("_______________________________Get List of Indices__________________________")
response = requests.get('http://unikge.cs.upb.de:5001/get-index-list')

j_obj = json.loads(response.text)
indices_list = j_obj['index_list']

print(response.text)

print("_____________________________________________________________________________")


# # ____________________________________________ Get Entity embedding by Label
# print("__________________________________ Get entity embedding by label ___________________________________________")
# json_data = {
#     'indexname': 'shallom_dbpedia_index',
#     'entities': ['resource/Giovanni_lombardi'],
# }
#
# response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding-bylabel', headers=headers, json=json_data)
# print(response.text)
# print("__________________________________ ___________________________________________")


# # ____________________________________________ Get Index information
# print("__________________________________ Get Index information ___________________________________________")
# json_data = {
#     'indexname': 'transe_dbpedia_l2_entity',
#     # 'indexname': 'shallom_dbpedia_index',
# }
# response = requests.get('http://unikge.cs.upb.de:5001/get-index-info', headers=headers, json=json_data)
#
# print(response.text)
# print("__________________________________ ___________________________________________")

# ________________________________ Getting all the entities of index
# print("_____________________________ Getting all the entities of index")
# json_data = {
#     # 'indexname': 'transe_dbpedia_l2_entity',
#     'indexname': 'shallom_dbpedia_index',
#     'size': 10
#
#
# }
#
# response = requests.get('http://unikge.cs.upb.de:5001/get-all-entity', headers=headers, json=json_data)
#
# print(len(json.loads(response.text)["entitylist"]))
# entitites = json.loads(response.text)["entitylist"]
# print()

# # _____________________________________________ Get entity embedding
# print("__________________________________ Get Entity Embedding ___________________________________________")
json_data = {
    # 'indexname': 'shallom_dbpedia_index',
    'indexname': 'transe_dbpedia_l2_entity',
    # 'entities': entitites,
    'entities': ['http://es.dbpedia.org/resource/Germany', 'http://dbpedia.org/resource/Patrick_Hayman'],
}

response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding', headers=headers, json=json_data)

answer = json.loads(response.text)
print(answer.keys())
# print(answer)
# emb = answer['http://es.dbpedia.org/resource/Germany']
# # emb = answer['/resource/Giovanni_Lombardi']
# print(len(emb))
# print("__________________________________ ___________________________________________")

# # _________________________________________________ Get Neighbour entities and their emb using ent_url
# print("__________________________________ Get Neighbout using URL___________________________________________")
# json_data = {
#     'indexname': 'shallom_dbpedia_index',
#     'entity': ['http://es.dbpedia.org/resource/Germany'],
#     'distmetric': 'cosine',
# }
#
# response = requests.get('http://unikge.cs.upb.de:5001/get-entity-neighbour', headers=headers, json=json_data)
#
# print(response.text)
# # answer = json.loads(response.text)
# # #
# # for k in answer['neighbours']:
# #     print(k)
# print("__________________________________ ___________________________________________")

# _______________________________________________ Get Neighbours from ent_emb
# print("__________________________________ Get Neigbour using ent_emb ___________________________________________")
# # print(str(emb).replace(",", ""))
# emb = str(emb).replace(" ", "")
#
# test = [0.287995577,-0.455794215,-0.560706854,-0.208061039,0.172987163,-0.582644463,-0.483516365,0.124980934,0.203231871,-0.656362116,-0.034981269,0.579211414,-0.180625692,0.278720349,0.065854639,0.306620687,-0.275009215,0.572021365,0.10284923,-0.658825278,0.203029677,-0.017125869,-0.261306912,-0.421797365,-0.29466486,-0.259765536,0.309641361,-0.353233963,0.741877079,-0.535071492,0.158671156,-0.173757523,-0.019785773,-0.245640799,-0.24233748,-0.488230139,0.354973942,-0.260450006,-0.008443747,0.295186102,-0.409880847,0.197315827,-0.198855579,-0.22186324,-0.411693335,0.740071476,-0.188669324,-0.077209555,0.047440559,0.084644906,0.101843677,-0.55739665,-0.086770803,0.17453979,-0.1399322,0.715744555,-0.784474909,-0.115671962,0.609704494,-0.167796418,0.011690577,0.250089586,-0.426775157,0.222675323,0.135895729,-0.602761865,-0.362928778,-0.293405741,0.300201267,0.058823615,0.415323079,0.391413987,-0.407318532,-0.318027765,0.032614678,0.179297313,-0.210481837,-0.191477463,0.034555513,-0.130783036,-0.411210477,0.798441648,0.285063356,0.636979163,0.089112148,-0.046494067,0.486533254,0.475836873,-0.070152186,-0.385979563,0.657763481,0.725894332,0.171367347,-0.161324874,0.351536393,-0.282844216,-0.287708282,-0.251460254,-0.344424963,-0.066946648]
#
# json_data = {
#     'indexname': 'transe_dbpedia_l2_entity',
#     'embedding': test,
#     'distmetric': 'cosine',
# }
#
# print(json_data)
# response = requests.get('http://unikge.cs.upb.de:5001/get-embedding-neighbour', headers=headers, json=json_data)
# print(response.text)
# print("__________________________________ ___________________________________________")
# #
# answer = json.load(response.text)
# for k in answer['Neighbour']:
#     print(k)
















# #
# # # ______________________________________________ Ordering a list of tuples according to the first element of the tuple
# #
# # f_data = '{"indexname":"shallom_dbpedia_index","embedding":[0.02233588,0.010766734,0.02364266,-0.027576402,0.07801491,0.042783223,-0.07689947,-0.079958074,-0.047613777,0.07463854,0.01335002,0.090599485,0.011700771,-0.07999231,0.011721943,-0.08457296,-0.021597078,0.011450011,-0.018370308,0.007592149,0.012584233,-0.10277818,0.057296358,-0.047838703,-0.008101291],"distmetric":"cosine"}'
# #
# # response = requests.get('http://unikge.cs.upb.de:5001/get-embedding-neighbour', headers=headers, data=data)
# #
# # print(response.text)
# #
# #
# # print(f_data == data)
# #
#
# # headers = {
# #     'Content-Type': 'application/json',
# # }
# #
# # emb = [0.287995577,-0.455794215,-0.560706854,-0.208061039,0.172987163,-0.582644463,-0.483516365,0.124980934,0.203231871,-0.656362116,-0.034981269,0.579211414,-0.180625692,0.278720349,0.065854639,0.306620687,-0.275009215,0.572021365,0.10284923,-0.658825278,0.203029677,-0.017125869,-0.261306912,-0.421797365,-0.29466486,-0.259765536,0.309641361,-0.353233963,0.741877079,-0.535071492,0.158671156,-0.173757523,-0.019785773,-0.245640799,-0.24233748,-0.488230139,0.354973942,-0.260450006,-0.008443747,0.295186102,-0.409880847,0.197315827,-0.198855579,-0.22186324,-0.411693335,0.740071476,-0.188669324,-0.077209555,0.047440559,0.084644906,0.101843677,-0.55739665,-0.086770803,0.17453979,-0.1399322,0.715744555,-0.784474909,-0.115671962,0.609704494,-0.167796418,0.011690577,0.250089586,-0.426775157,0.222675323,0.135895729,-0.602761865,-0.362928778,-0.293405741,0.300201267,0.058823615,0.415323079,0.391413987,-0.407318532,-0.318027765,0.032614678,0.179297313,-0.210481837,-0.191477463,0.034555513,-0.130783036,-0.411210477,0.798441648,0.285063356,0.636979163,0.089112148,-0.046494067,0.486533254,0.475836873,-0.070152186,-0.385979563,0.657763481,0.725894332,0.171367347,-0.161324874,0.351536393,-0.282844216,-0.287708282,-0.251460254,-0.344424963,-0.066946648]
# # s2 = f"{emb}"
# # s3 = ""
# # for i in s2:
# #     if i != " ":
# #         s3 += i
# #
# #
# # f_data = '{"indexname":"shallom_dbpedia_index","embedding":[0.02233588,0.010766734,0.02364266,-0.027576402,0.07801491,0.042783223,-0.07689947,-0.079958074,-0.047613777,0.07463854,0.01335002,0.090599485,0.011700771,-0.07999231,0.011721943,-0.08457296,-0.021597078,0.011450011,-0.018370308,0.007592149,0.012584233,-0.10277818,0.057296358,-0.047838703,-0.008101291],"distmetric":"cosine"}'
# # data = f"{{\"indexname\":\"shallom_dbpedia_index\",\"embedding\":{s3},\"distmetric\":\"cosine\"}}"
# #
# #
# # print(data == f_data)
# # import requests
# # response = requests.get('http://unikge.cs.upb.de:5001/get-embedding-neighbour', headers=headers, data=data)
# # print(response.text)
# # print("__________________________________ ___________________________________________")
#
# # from elasticsearch import Elasticsearch
# # es = Elasticsearch(["http://unikge.cs.upb.de:9200"])
# #
# # print(es.indices.get_alias(index="*"))
#
# # print(es.info())
#
# import requests
# from Data_Manager import *
# import json
# import csv
#
#
# headers = {
#     # Already added when you pass json= but not when you pass data=
#     # 'Content-Type': 'application/json',
# }
#
# log("Organizing Data in a dictionary...", 0)
# my_py_index = prep_data_without_es("./../Data/Results/mentions_embeddings_with_doc_id.tsv")
#
#
# t_count = 0
# p_count = 0
# n_count = 0
#
# neg_file = open("./../Data/Results/negatives.txt", "w", encoding="utf8")
# with open("./../Data/Results/entity_mentions.tsv", "w", encoding='utf8') as out_f:
#     writer = csv.writer(out_f, delimiter="\t")
#     written = []
#
#     for tid in my_py_index:
#         print(f"text ===> {tid}")
#         for key in my_py_index[tid]:
#
#             x = key[1].replace("http://dbpedia.org", "")
#
#             json_data = {
#                 'indexname': 'shallom_dbpedia_index',
#                 'entities': [
#                     x,
#                 ],
#             }
#
#             t_count += 1
#
#             response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding', headers=headers, json=json_data)
#
#             if "404" in response.text:
#                 n_count += 1
#                 neg_file.write(f"{key[1]}\n")
#                 print(f"Error ====> {x}")
#             else:
#                 if key[1] not in written:
#                     try:
#                         p_count += 1
#                         ans = json.loads(response.text)
#                         emb = ans[x]
#                         writer.writerow([key[1], emb])
#                         written.append(key[1])
#                     except json.decoder.JSONDecodeError:
#                         neg_file.write(f"{key[1]}\n")
#                         n_count += 1
#                         print(response.text)
#                         print("____________________")
#
#     print(f"Total count = {t_count}, p_count = {p_count}, n_count = {n_count}")
# import csv
# existent = []
# with open("./../Data/Results/entity_mentions.tsv", "r", encoding='utf8') as emf:
#     reader = csv.reader(emf, delimiter='\t')
#
#     for row in reader:
#         if len(row) == 2:
#             existent.append(row[0])
#
# emf.close()
#
# retry = []
# with open("./../Data/Results/negatives.txt", "r", encoding='utf8') as nf:
#     for line in nf.readlines():
#         if line in existent:
#             continue
#         else:
#             retry.append(line)
# nf.close()
#
# import requests
# from Data_Manager import *
# import json
# import csv
#
# t_count = 0
# p_count = 0
# n_count = 0
#
# headers = {
#     # Already added when you pass json= but not when you pass data=
#     # 'Content-Type': 'application/json',
# }
#
# log("Organizing Data in a dictionary...", 0)
# my_py_index = prep_data_without_es("./../Data/Results/mentions_embeddings_with_doc_id.tsv")
#
#
# with open("./../Data/Results/entity_mentions(2).tsv", "w", encoding='utf8') as out_f:
#     writer = csv.writer(out_f, delimiter="\t")
#     written = []
#
#     for key in retry:
#
#         x = key.replace("http://dbpedia.org", "")
#
#         json_data = {
#             'indexname': 'shallom_dbpedia_index',
#             'entities': [
#                 x,
#             ],
#         }
#
#         t_count += 1
#
#         response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding', headers=headers, json=json_data)
#
#         if "404" in response.text:
#             n_count += 1
#             print(f"Error ====> {x}")
#         else:
#             if key[1] not in written:
#                 try:
#                     p_count += 1
#                     ans = json.loads(response.text)
#                     emb = ans[x]
#                     writer.writerow([key[1], emb])
#                     written.append(key[1])
#                 except json.decoder.JSONDecodeError:
#                     n_count += 1
#                     print(response.text)
#                     print("____________________")
#
# print(f"Total count = {t_count}, p_count = {p_count}, n_count = {n_count}")

# import csv
#
# import torch
# from torch.utils.data import DataLoader
#
#
# # self.e_emb = []
# # self.m_emb = []
# # self.m_emb.append(d_entry[2])
# # self.e_emb.append(d_entry[3])
#
#
# class EmbeddingsMyDataset:
#     def __init__(self, data2):
#         self.x = []
#         self.y = []
#
#         for d_entry in data2:
#             self.x.append((d_entry[0], d_entry[2]))
#             self.y.append((d_entry[1], d_entry[3]))
#
#         self.n_samples = len(self.x)
#
#     def __getitem__(self, index):
#         return self.x[index], self.y[index]
#
#     def __len__(self):
#         return self.n_samples
#
#
# en_file = open("./../Data/Results/transe_entity_embeddings.tsv", encoding='utf8')
# en_reader = csv.reader(en_file, delimiter="\t")
#
# e_emb_set = {}
#
# for row in en_reader:
#     if len(row) == 2:
#         e_emb_set[row[0]] = row[1]
# en_file.close()
#
# k = 0
#
# men_file = open("./../Data/Results/mentions_embeddings_with_doc_id.tsv", encoding="utf8")
# reader = csv.reader(men_file, delimiter="\t")
#
# data = []
# for row in reader:
#
#     mention = row[2]
#     entity = row[1]
#
#     try:
#         emb = e_emb_set[row[1]]
#         e_emb = torch.FloatTensor([float(elem) for elem in emb.strip("[] ").split(",")])
#         if e_emb.size()[0] != 25:
#             print(f"e_emb = {e_emb.size()[0]}")
#         en_found = True
#
#     except KeyError:
#         en_found = False
#
#     if en_found:
#         m_emb = []
#         for elem in row[3].strip('[]\n').split(" "):
#             try:
#                 m_emb.append(float(elem))
#             except ValueError:
#                 pass
#         m_emb = torch.FloatTensor(m_emb)
#         if m_emb.size()[0] != 768:
#             print(f"m_emb = {m_emb.size()}")
#     else:
#         continue
#
#     data.append((mention, entity, m_emb, e_emb))
#
#     if k == 24:
#         break
#     k += 1
#
#
# dataset = EmbeddingsMyDataset(data)
# dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True, num_workers=0)
#
#
# print(dataset.__len__())
#
#
# for i, x in enumerate(dataloader):
#     print(x[0][0])
#     print(x[1][0])
#     break

# import csv
# from collections import Counter
#
# count = {}
#
# file = open("../Data/Results/train_data.tsv", "r", encoding="utf-8")
# reader = csv.reader(file, delimiter="\t")
#
# for row in reader:
#     if len(row) > 1:
#         eg = row[1]
#
#         if eg in count.keys():
#             count[eg] += 1
#         else:
#             count[eg] = 1
#
#
# c = Counter(count)
# s = []
# for k in c.most_common():
#     s.append(k)
#
# print(len(s))
#
# count = 0
# for e in s:
#     if e[1] == 1:
#         continue
#     else:
#         count += 1
#
#
# print(count)
# import torch
# import torch.nn as nn
#
# t = torch.FloatTensor([1, 3, 4, 5, 6])
# t.requires_grad = True
# t2 = torch.FloatTensor([200000, 2000000, 3000000, 43287372, 523849])
#
# c = nn.MSELoss()
#
# loss = c(t, t2)
# print(loss)
#
# rloss = torch.sqrt(c(t, t2))
# print(rloss)
#
# rloss.backward()
# print(t.grad)
#
# print(t)

