# import requests
# import json
#
# response = requests.get('http://unikge.cs.upb.de:5001/get-index-list')
#
# j_obj = json.loads(response.text)
# indices_list = j_obj['index_list']
#
# headers = {
#     # Already added when you pass json= but not when you pass data=
#     # 'Content-Type': 'application/json',
# }
#
# print(indices_list)
#
# for ind_name in indices_list:
#     json_data = {
#         'indexname': 'shallom_dbpedia_index',
#         'size': 10000,
#     }
#     response = requests.get('http://unikge.cs.upb.de:5001/get-all-entity', headers=headers, json=json_data)
#
#     try:
#         res_obj = json.loads(response.text)
#
#         print(res_obj)
#
#     except json.decoder.JSONDecodeError as j_err:
#         print(response.status_code)
#

import torch


z = torch.rand((1, 5))
t = torch.ones((7, 5))

r = torch.split(t, 1, 0)
print(len(r))
print(r)
