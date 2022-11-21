# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME

from Code.Logger import log

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from Code.blink.common.ranker_base import BertEncoder, get_model_obj
from Code.blink.common.optimizer import get_bert_optimizer


from Code.blink.common.optimizer import get_bert_optimizer


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])

        self.kge_emb_dim = params["kge_emb_dim"]
        self.bert_output_dim = params["out_dim"]

        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )

        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            kge_emb_dim=params["kge_emb_dim"]
        )
        self.cand_additional_layer = nn.Linear(self.kge_emb_dim+self.bert_output_dim, self.bert_output_dim)

        self.dropout = nn.Dropout(0.1)
        self.config = ctxt_bert.config

    def forward(self, token_idx_ctxt, segment_idx_ctxt, mask_ctxt, token_idx_cands, segment_idx_cands, mask_cands,
                kge_emb=None,):
        # Embedding context of the mention
        embedding_ctxt = None
        if token_idx_ctxt is not None:
          embedding_ctxt = self.context_encoder(
                                  token_idx_ctxt, segment_idx_ctxt, mask_ctxt
                          )

        # Embedding context of the entity
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )

            if kge_emb is not None:
                embeddings = torch.cat((embedding_cands, kge_emb), 1)
                embedding_cands = self.cand_additional_layer(self.dropout(embeddings))

        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        log("Setting the parameters fot the BiEncoderRanker...", 1)
        super(BiEncoderRanker, self).__init__()
        self.params = params
        #self.device = torch.device(
        #    "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        #)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = "cpu"
        self.n_gpu = torch.cuda.device_count()

        # init tokenizer
        log("Adding tokens to the tokenizer...", 1)
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
        # init model
        log("Building the model... (actual bi-encoder)", 1)
        self.build_model()

        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        log(f"sending model to device {self.device}", 1)
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"\t Loaded the model weight from {fname}")

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
#        return embedding_context.cpu().detach()
        return embedding_context.to(self.device)

    def encode_candidate(self, cands, kge_emb=None):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        if kge_emb is not None:
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands, kge_emb=kge_emb
            )
        else:
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )
#        return embedding_cands.cpu().detach()
        return embedding_cands.to(self.device)
        # TODO: why do we need cpu here?
        # return embedding_cands

    def compute_new_embeddings(self, model_output, kge):
        input_emb = torch.cat((model_output, kge), dim=2).to(self.device)
        out = self.model.cand_additional_layer(self.model.dropout(input_emb))

        return out

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(self, text_vecs, cand_vecs, random_negs=True, cand_encs=None, kge_emb=None):
        # try:
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embeddings.bmm(torch.FloatTensor(cand_vecs))
#            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands, kge_emb=kge_emb
        )
        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, kge_emb=None, label_input=None):

        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag, kge_emb=kge_emb)
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        return loss, scores

    def compute_loss(self, context_encs, cand_encs, labels):
        # train on hard negatives
        embedding_ctxt = context_encs.unsqueeze(1)  # batchsize x 1 x embed_size
        embedding_cands = cand_encs.permute(0, 2, 1)  # batchsize x embed_size x 2
        scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
        scores = torch.squeeze(scores)

        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
        # TODO: add parameters?
        loss = loss_fct(scores, labels.to(self.device))

        return loss, scores

    def add_layer(self):
#        self.tokenizer.add_tokens(['[m]', '[/m]', '[ENT]'])
#        self.model.add_layer(len(self.tokenizer))
        pass


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()

    return token_idx.clone().detach(), segment_idx.clone().detach(), torch.tensor(mask).to(torch.int64)
