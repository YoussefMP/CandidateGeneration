from elasticsearch import Elasticsearch
import requests


# TODO: Introduce methods for Transforming curl calls to http requests
#                         for Processing the response of the server


class IndexManager:
    def __init__(self, base_url):
        # if knn:
        #     # self.es_instance = Elasticsearch(hosts=host_id, basic_auth=('elastic', key), request_timeout=60)
        #     # self.knn_wrapper = ElastiknnClient(self.es_instance)
        #     # self.client = self.knn_wrapper.es
        #     pass
        # else:
        #     self.client = Elasticsearch(cloud_id=host_id,
        #                                 basic_auth=("elastic", key))

        self.base_url = base_url

    def send_request(self, url_ext, header, data):
        url = self.base_url + "/" + url_ext
        response = requests.get(url=url, header=header, data=data)

        return response

    def create_index(self, index_name, file_path, dim):
        print(self.base_url)
        return False

