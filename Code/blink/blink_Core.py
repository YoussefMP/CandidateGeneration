from transformers import BertTokenizer, BertModel, WEIGHTS_NAME, CONFIG_NAME
from Code.blink.biencoder import BiEncoderModule
from Code.Logger import log
from torch import optim, nn
import numpy as np
import torch
import time

import os

__DEBUG__ = True
DEBUG_SIZE = 8
DEBUG_INDEX_SIZE = 100

def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    attention_mask = []
    for dim in range(len(token_idx)):
        attention_mask.append([])
        for i in token_idx[dim]:
            if i.item() == 0:
                attention_mask[-1].append(0)
            else:
                attention_mask[-1].append(1)
    attention_mask = torch.IntTensor(attention_mask)

    segment_idx = torch.zeros(token_idx.size()).type(torch.IntTensor)
    return token_idx, attention_mask, segment_idx


def do_pass(model, dataloader):
    for i, data in enumerate(dataloader):

        mention = data[0][1]
        source = data[0][2]

        entity = data[1][0]
        target = data[1][1]

        source, attention_mask, segment_idx = to_bert_input(source, 0)
        t, s = model(source, attention_mask, segment_idx, None, None, None)

        target, attention_mask, segment_idx = to_bert_input(target, 0)
        tt, st = model(None, None, None, target, attention_mask, segment_idx)

        if t is not None:
            print(f"{t.size()} ===> {t}")

        if s is not None:
            print(f"{s.size()} ===> {s}")


def extract_mentions_context(text, s_token, e_token,
                             cls, pad, sep):
    contexts = []

    preceding_context = []
    succeeding_context = []

    in_mention = False
    for tid in range(len(text)):
        token = text[tid]

        if in_mention:
            preceding_context.append(token)

        if token == s_token:
            preceding_context = text[tid - 32: tid + 1] if tid - 32 >= 0 else text[:tid + 1]
            in_mention = True
        elif token == e_token:
            succeeding_context = text[tid + 1:tid + 32] if tid + 32 < len(text) else text[tid + 1:]
            in_mention = False

            context = [cls] + \
                      preceding_context + \
                      succeeding_context + \
                      [sep] + \
                      [pad] * (76 - len(preceding_context + succeeding_context))

            contexts.append(context)

    return contexts


def prepare_data(file_manager, file, file_type):

    txt_files_name = f"annotated_text_" + file + ".txt"

    log(f"[BLINK] Reading {file} data file ...", 1)
    file_manager.add_path(file_type, file, "Datasets")

    debug_size = DEBUG_SIZE if __DEBUG__ else None
    texts, ordered_golden_entities, ordered_mentions = file_manager.read_aida_input_doc(file_type, debug_size=debug_size)

    file_manager.write_lines(txt_files_name, texts.values())

    return texts, ordered_golden_entities, ordered_mentions


class BiEncoder:
    def __init__(self, bi_encoder=None, file_manager=None):
        # Object for handling fileIO operations
        self.file_manager = file_manager

        # Loading the tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Adding mention tags to the tokenizers vocabulary
        self.bert_tokenizer.add_tokens('[m]')
        self.bert_tokenizer.add_tokens('[/m]')

        # Initializing the bi-encoder from the BLINK project
        self.bi_encoder = bi_encoder
        self.bi_encoder.ctxt_bert.resize_token_embeddings(len(self.bert_tokenizer))

        # Dictionary containing all entities of the entity embeddings index
        self.entity_embeddings_index = None
        self.hot_index = {}
        # FAISS index for the new computed entity embeddings
        self.faiss_index = None

    def load_enti_emb(self):
        self.entity_embeddings_index = self.file_manager.load_ent_embedding()

    def warm_up(self, file, file_type):
        """
        Prepares the entity, mention embeddings pairs for the training loop.
        :param file: file name, from which the data will be read and loaded
        :param file_type: types differentiate between training and validation
        :return:
        """

        # Get annotated texts and the list of mentions and entities in the file
        texts, ordered_golden_entities, mentions = prepare_data(self.file_manager, file, file_type)
        ordered_mention_contexts = {}

        # Load the entity embeddings in a dictionary and extract the one we found in our data
        self.load_enti_emb()
        ordered_entity_embeddings = {}
        for k, v in ordered_golden_entities.items():
            if not v:
                ordered_entity_embeddings[k] = []
            for entity in v:
                if ordered_entity_embeddings.get(k):
                    try:
                        ordered_entity_embeddings[k].append((entity, self.entity_embeddings_index[entity]))
                    except KeyError:
                        ordered_entity_embeddings[k].append(None)
                else:
                    try:
                        ordered_entity_embeddings[k] = [(entity, self.entity_embeddings_index[entity])]
                    except:
                        ordered_entity_embeddings[k] = [None]

        # Get mentions contexts maximal length is 78 => 32 word pieces + mention + 32 word pieces
        for text in range(len(texts)):
            tokenized_text = self.bert_tokenizer.tokenize(texts[text])
            text_token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)

            s_tag = self.bert_tokenizer.convert_tokens_to_ids('[m]')
            e_tag = self.bert_tokenizer.convert_tokens_to_ids('[/m]')

            pad = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
            cls = self.bert_tokenizer.convert_tokens_to_ids('[CLS]')
            sep = self.bert_tokenizer.convert_tokens_to_ids('[SEP]')
            ordered_mention_contexts[text] = extract_mentions_context(text_token_ids, s_tag, e_tag,
                                                                      cls, pad, sep)

        return ordered_mention_contexts, ordered_entity_embeddings

    def training_loop(self, dataloader, epochs, learning_rate):

        # Training Hyper parameters
        time_sum = 0
        log("Starting the training ...", 1)
        step = 0

        # Best epoch performance index
        best_epoch_idx = -1
        best_epoch_weights = None
        min_epoch_loss = np.inf

        # Loss function and optimizer
        optimizer = optim.Adam(self.bi_encoder.parameters(), lr=learning_rate)
        criterion2 = nn.CosineSimilarity()
        criterion = nn.MSELoss()

        log("Entering Epoch loop", 1)
        # Training Loop
        for epoch in range(epochs):

            e_start = time.time()
            log(f"____ Going through Epoch {epoch} ________ ", 1)

            # set model in training mode
            self.bi_encoder.train()
            losses = []

            for idx, data in enumerate(dataloader):

                mention_input_ids = data[0]
                target_embedding = data[1][1]

                # Input: [batch_size, inputs_dimensions]
                token_idx, attention_mask, segment_idx = to_bert_input(mention_input_ids, 0)
                ctxt_outputs = self.bi_encoder(mention_input_ids, attention_mask, segment_idx, None, None, None)[0]

                target_input, attention_mask, segment_idx = to_bert_input(target_embedding, 0)
                ent_output = self.bi_encoder(None, None, None, target_input.unsqueeze(1), attention_mask, segment_idx)[1]

                optimizer.zero_grad()

                loss = criterion(ctxt_outputs, ent_output)
                loss_2 = criterion2(ctxt_outputs, ent_output)
                loss_2 = -loss_2 + 1
                loss = loss + loss_2.mean()
                loss.sum().backward()

                torch.nn.utils.clip_grad_norm_(self.bi_encoder.parameters(), max_norm=1)
                optimizer.step()

                step += 1

                losses.append(loss)

                if (idx % 10) == 0 and idx > 1:
                    e_end = time.time()
                    sid = 0
                    log(f"Epoch_{epoch}: is {(idx * 100) / dataloader.__len__()}% Done,"
                        f" achieved loss = {sum(losses) / len(losses)},"
                        f" time for batch process: {e_end - e_start}", 2)

            e_end = time.time()
            log(f"Epoch_{epoch}: Avg-Loss = {sum(losses) / len(losses)}, Epoch lasted: {e_end - e_start}", 0)
            # log(f"Epoch_{epoch}: Avg-Loss = {sum(losses) / len(losses)}, Epoch lasted: {e_end - e_start}", 0)
            time_sum += e_end - e_start

            if min_epoch_loss > (sum(losses) / len(losses)):
                min_epoch_loss = (sum(losses) / len(losses))
                best_epoch_idx = epoch

        return time_sum

    def generate_faiss_entity_index(self):

        self.bi_encoder.eval()

        new_embeddings_index = []
        eid = 0

        if self.entity_embeddings_index is None:
            self.load_enti_emb()

        if __DEBUG__:
            keys = list(self.entity_embeddings_index.keys())[:100]
        else:
            keys = list(self.entity_embeddings_index.keys())

        for entity in keys:
            embedding = self.entity_embeddings_index[entity].unsqueeze(0)
            embedding, attention_mask, segments_idx = to_bert_input(embedding, 0)

            with torch.no_grad():
                outputs = self.bi_encoder(None, None, None, embedding.unsqueeze(0), attention_mask, segments_idx)

            self.hot_index[eid] = (entity, outputs[1])
            eid += 1
            new_embeddings_index.append(np.array(outputs[1].squeeze(0)))

        import faiss

        self.faiss_index = faiss.IndexFlatIP(25)
        print(self.faiss_index.is_trained)

        self.faiss_index.add(np.array(new_embeddings_index))

    @staticmethod
    def save_model_state(model, output_dir, model_name):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = model.module if hasattr(model, "module") else model
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)


def main(file_manager, params=None):
    print("setting parameters for the module")
    if not params:
        params = {
            "out_path": "./../../Models/blink",
            "no_cuda": True,
            "bert_model": "bert-base-uncased",
            "path_to_model": None,
            "data_parallel": False,
            "out_dim": 25,
            "pull_from_layer": 1,
            "add_linear": True,
        }

    print(f"initiating module with: {params}")
    module = BiEncoder(bi_encoder=BiEncoderModule(params), file_manager=file_manager)

    return module
