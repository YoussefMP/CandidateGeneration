import sys

from transformers import BertTokenizer, BertModel
import os
import re


class ModelFileIO:
    def __init__(self, f_path):
        self.paths = {'input': f_path}

    def add_path(self, f_type, f_path):
        self.paths[f_type] = f_path

    def read_input_doc(self):

        relations_set = {}
        entities_set = {}

        relation_regex = re.compile(r'''<a href="(?P<href>.*)" title="(?P<entity>.*"?)" relation="(?P<relation>.*)"''')
        entity_regex = re.compile(r'''<a href="(?P<href>.*)" title="(?P<entity>.*"?)"''')

        # Loading the entities and relations set
        input_doc = open(self.paths['input'])

        input_doc_lines = input_doc.readlines()
        for line in input_doc_lines:
            line = line.strip("\n")
            if line.startswith("url"):
                arg1 = line[line.rfind('/')+1:].replace("_", " ")
                entities_set[arg1] = line.strip('url=')
            elif line:
                related_entities = []
                data_blocks = line.split("</a>")
                for block in data_blocks:
                    match_relation = re.search(relation_regex, block)
                    if match_relation:
                        related_entities.append((match_relation.group('relation'), match_relation.group('entity')))
                        entities_set[match_relation.group('entity')] = match_relation.group('href')
                    else:
                        match_entity = re.search(entity_regex, block)
                        if match_entity:
                            entities_set[match_entity.group('entity')] = match_entity.group('href')

                relations_set[arg1] = related_entities

        return relations_set, entities_set


class Translator:
    def __init__(self):
        # Loading the tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Loading bert pretrained model
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)


if __name__ == "__main__":
    # setting input_doc path
    file_path = "./../Datasets/wikipedia.test"
    file_manager = ModelFileIO(file_path)

    # Extracting data from file
    relations, entities = file_manager.read_input_doc()


