from transformers import BertTokenizer, BertModel
from Logger import log
import torch
import re
import os


M_TAG_LENGTH = 9


def truncate_at_last_mention(text):
    t_res, id_res = [], 0

    text.reverse()
    idx = text.index("[/m]")
    ridx = len(text) - idx

    text.reverse()

    t_res, id_res = text[:ridx], ridx
    if len(t_res) > 254:
        t_res, id_res = truncate_at_last_mention(text[:ridx-1])

    return t_res, id_res


def split_into_chunks(tokenized_text):
    in_mention = False
    chunks = []
    idx = 0
    offset = 0

    for tid in range(len(tokenized_text)):
        if (tokenized_text[tid] == "." and not in_mention and tid - offset < 254) or tid == len(tokenized_text)-1:
            idx = tid

        if tid - offset > 254 or tid == len(tokenized_text) - 1:
            if idx + 1 <= offset or (len(tokenized_text[offset:idx + 1]) > 254 and in_mention) or \
                    (len(tokenized_text[offset:idx + 1]) > 254 and tid == len(tokenized_text) - 1):
                last_chunk, local_offset = truncate_at_last_mention(tokenized_text[offset:tid])
                last_chunk = ["[CLS]"] + last_chunk
                offset += local_offset

            else:
                last_chunk = ["[CLS]"] + tokenized_text[offset:idx + 1]
                offset = idx + 1

            padding = ["[PAD]" for _ in range(255 - len(last_chunk))]
            last_chunk += padding
            last_chunk.append("[SEP]")
            chunks.append(last_chunk)

        if tokenized_text[tid] == "[m]":
            in_mention = True
        elif tokenized_text[tid] == "[/m]":
            in_mention = False

    return chunks


def compute_word_embedding(span, text_emb):
    """
    :param span: [int, ...] indexes of tokens that form the mention surface form
    :param text_emb: the embedding of the text; shape = []
    :return:
    """
    vectors = []
    for token_id in span:
        vectors.append(torch.sum(text_emb[token_id][-4:], dim=0))

    mention_embedding = torch.stack(vectors, dim=0).mean(0)

    return mention_embedding


def generate_ent_list():
    # TODO: Change path to your path of the index of entity embeddings
    path = "../Data/Results/transe_entity_embeddings.tsv"
    if os.path.exists(path):
        import csv

        ent_list = []
        data = open(path, 'r', encoding='utf8')
        reader = csv.reader(data, delimiter='\t')

        for row in reader:
            if len(row) == 2:
                ent_list.append(row[0])

        return ent_list
    return False


class Encoder:
    def __init__(self):
        # Loading the tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Adding mention tags to the tokenizers vocabulary
        self.bert_tokenizer.add_tokens('[m]')
        self.bert_tokenizer.add_tokens('[/m]')

        # Loading bert pretrained model
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        # Updating the vocab size
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        # Put the model in evaluation mode
        self.bert_model.eval()

    def get_mentions_spans(self, ids):
        """
        Given a list of token_ids this computes and returns the spans of *all* the mentions in the text.
        :param ids: list of token_ids
        :return:
        """
        mentions = []
        m_span = []
        last = True
        it = 0

        s_tag = self.bert_tokenizer.convert_tokens_to_ids('[m]')
        e_tag = self.bert_tokenizer.convert_tokens_to_ids('[/m]')

        try:
            while it < len(ids):
                if ids[it] == s_tag:
                    it += 1
                    while ids[it] != e_tag:
                        m_span.append(it)
                        it += 1
                        if it == len(ids):
                            last = False
                            break
                    if last:
                        mentions.append(m_span)
                        m_span = []
                else:
                    it += 1
        except IndexError:
            print("tex")
            pass

        return mentions

    def embed_mentions(self, corpus, ordered_golden_entities, ordered_mentions):
        """
        Given an annotated corpus this methods generates embeddings for each tagged mention in the text
        :param 
        corpus: List of annotated texts
        ordered_golden_entities: List of golden entities sorted as in the order in which they will be processed
        ordered_mentions: List of mentions sorted in the order in which they appear in the text
        :return: text_mentions_embeddings: {{text_id}} : {{mention}}: emb}}
        """
        # text_mentions_embeddings = {}
        text_mentions_embeddings = []
        corpus_chunks = {}
        eg_emb_list = generate_ent_list()

        # Iterating the list of annotated texts
        for text_id in list(corpus.keys()):
            log(f"processing Text{text_id} / {len(list(corpus.keys()))}", 1, out=text_id % 20 == 0 or text_id == 99)

            tokenized_text = self.bert_tokenizer.tokenize(corpus[text_id])
            tokenized_text = split_into_chunks(tokenized_text)
            corpus_chunks[text_id] = tokenized_text

            for chunk in range(len(tokenized_text)):
                # Generate list of ids out of the tokenized text
                indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text[chunk])
                # Converting inputs from list to tensors
                tokens_tensor = torch.tensor([indexed_tokens])

                mentions_spans = self.get_mentions_spans(indexed_tokens)

                # Segment tensor
                segments_tensor = torch.ones(256)
                segments_tensor = segments_tensor.unsqueeze(0)

                with torch.no_grad():
                    # Encoding the input text
                    outputs = self.bert_model(tokens_tensor, segments_tensor)
                    hidden_states = outputs[2]
                    # Extracting the hidden layers. "Representation of text"
                    sentence_embedding = hidden_states

                    # Reshaping the hidden layers.
                    text_emb = torch.stack(sentence_embedding, dim=0)
                    text_emb = torch.squeeze(text_emb, dim=1)
                    text_emb = text_emb.permute(1, 0, 2)

                    # if not text_mentions_embeddings.get(text_id):
                    #     entity_mention_pairs = {}
                    #     text_mentions_embeddings[text_id] = entity_mention_pairs

                    # pooling function. Averaging the last four layers.
                    for m_span in mentions_spans:
                        golden_entity = ordered_golden_entities[text_id].pop(0)
                        mention_sf = ordered_mentions[text_id].pop(0)

                        if eg_emb_list and (golden_entity not in eg_emb_list):
                            # print(f"skipped {mention_sf} because {golden_entity} is not in the eg_index")
                            continue

                        emb = compute_word_embedding(m_span, text_emb)
                        text_mentions_embeddings.append(((text_id, chunk), golden_entity, mention_sf, emb))

        return text_mentions_embeddings, corpus_chunks

