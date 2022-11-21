# ________________________________ Testing My ElasticSearch Instance ____________________________________________
# from elasticsearch import Elasticsearch, helpers
#
#
# es = Elasticsearch(hosts="https://datasets-26efe5.es.us-central1.gcp.cloud.es.io:9243",
#                    basic_auth=('elastic', "CisJmi0r6jd7oIWqPFdB21Av"), request_timeout=60)
#
# response = helpers.scan(es, index="entity_embeddings", query={"query": {"match_all": {}}})
#
# count = 0
# for res in response:
#     print(len(res['_source']['embeddings']))
#     break
# print(count)

# ________________________________________________________________________________________________________________
# ________________________________ Splitting Tuples in an array in tow datastructures ________________________________
# a = []
# for i in range(10):
#     j = (5 * i + 10) / 2
#     a.append((i, j))
#
# b = [x[0] for x in a]
# c = [x[1] for x in a]
#
# print(b)
# print(c)
# ________________________________________________________________________________________________________________
# __________________________________________ Miscellaneous ________________________________________________________
import torch

inps = torch.FloatTensor([-0.5602, -1.5577])

d = torch.FloatTensor([1])

c = torch.cat((d, inps), dim=0)

print(c, end=f"====> {c.size()}")
print(inps.size())
print(inps[0])

