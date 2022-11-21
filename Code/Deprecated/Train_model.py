from transformers import EncoderDecoderModel, BertTokenizer
from Indices_Manager import IndexManager
from elasticsearch import helpers
from torch.optim import AdamW
from torch import nn
import numpy as np
import random
import torch
import math
import time

__DEBUG__ = True


def prep_train_data(es):
    """
    :param es: Elasticsearch client manager.
    :return: List of tuples of embeddings connecting mentions to entities [([float], [float]), (), ...]
    """

    pairs_generated = 0

    training_pairs = []
    response = helpers.scan(es.client, index="entity_mentions", query={"query": {"match_all": {}}})
    for doc in response:
        ent_lookup = es.client.search(index="entity_embeddings",
                                      query={"match": {
                                          "entity": doc['_source']['entity'][doc['_source']['entity'].rfind('/'):]
                                      }})
        for hit in ent_lookup['hits']['hits']:
            if hit['_source']['entity'] == doc['_source']['entity']:
                entity_embedding = hit["_source"]["embeddings"]

                for mention in doc['_source']['mentions']:
                    mention_emb_lookup = es.client.search(index="mention_embeddings",
                                                          query={"match": {
                                                              "mention": mention
                                                          }})
                    for m_hit in mention_emb_lookup['hits']['hits']:
                        if m_hit['_source']['mention'] == mention.lower():
                            training_pairs.append((entity_embedding, m_hit['_source']['embeddings']))
        pairs_generated += 1
        if pairs_generated == 2 and __DEBUG__:
            return training_pairs

        if pairs_generated % 20 == 0:
            print(f"{(pairs_generated*100)/15909892}% Done")

    return training_pairs


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


class SequenceToSequenceModel:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # initialize Bert2Bert from pre-trained checkpoints
        self.seq2seq_model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased',
                                                                                 'bert-base-uncased')

        self.seq2seq_model.config.decoder_start_token_id = self.bert_tokenizer.cls_token_id
        self.seq2seq_model.config.pad_token_id = self.bert_tokenizer.pad_token_id
        self.seq2seq_model.config.vocab_size = self.seq2seq_model.config.decoder.vocab_size + 2

        self.softmax = nn.LogSoftmax(dim=2)

    def train_seq2seq_model(self, criterion, training_pairs, epochs, learning_rate=0.01):
        start = time.time()
        self.seq2seq_model.train()

        # model_optimizer = optim.SGD(s_model.parameters(), lr=learning_rate)
        model_optimizer = AdamW(self.seq2seq_model.parameters(),
                                lr=2e-5,
                                eps=1e-8)
        loss = 0

        for epoch in range(epochs):
            random.shuffle(training_pairs)

            print(f'Epoch {epoch}: {time_since(start, epoch + 1 / epochs)}, \t Loss = {loss if loss else 0}\n\t\t')

            p = 0
            epoch_min_loss = np.inf
            saved_vals = []

            for pair in training_pairs:
                attention_mask = []

                input_ids = torch.FloatTensor(pair[0]).unsqueeze(0)
                labels = torch.FloatTensor(pair[1]).unsqueeze(0)

                print(f"pair[0] = {input_ids.size()}")
                print(f"pair[1] = {labels.size()}")

                model_optimizer.zero_grad()
                outputs = self.seq2seq_model(input_ids=input_ids, labels=labels)

                logits = outputs.logits
                loss = outputs.loss

                loss.backward()

                model_optimizer.step()

                if epoch_min_loss > loss:
                    query_prediction = self.softmax(logits)
                    _, tokenized_prediction = torch.max(query_prediction, 2)

                    epoch_min_loss = loss

                if p % 5 == 0 or p == 31:
                    percent = ((p + 1) * 100) / 32
                    print(f"==> epoch is {percent:.2f}% done \t\t Loss achieved is loss = {loss}")
                p += 1

        return outputs


if __name__ == "__main__":

    print("Starting Elasticsearch instance... ")
    es_instance_manager = IndexManager(host_id="https://datasets-26efe5.es.us-central1.gcp.cloud.es.io:9243",
                                       key="CisJmi0r6jd7oIWqPFdB21Av")

    print("Prepping the training pairs")
    training_data = prep_train_data(es_instance_manager)

    Epochs = 1000
    Criterion = nn.MSELoss()

    model = SequenceToSequenceModel()
    model.train_seq2seq_model(Criterion, training_data, Epochs)

