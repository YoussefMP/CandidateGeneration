from transformers import BertTokenizer, BertModel
import os


class ModelFileIO:
    def __init__(self, f_path):
        self.paths = {'input': f_path}

    def add_path(self, f_type, f_path):
        self.paths[f_type] = f_path

    def read_input_doc(self):
        # Loading the entities and relations set
        input_doc = open(self.path)

        input_doc_lines = input_doc.readlines()
        for line in input_doc_lines:


class Translator:
    def __init__(self):
        # Loading the tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Loading bert pretrained model
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)


if __name__ == "__main__":
    # setting input_doc path
    file_path = "./Datasets/wikipedia.test"
    file_manager = ModelFileIO(file_path)


