# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
from Code.Logger import log


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


class BertEncoder(nn.Module):
    def __init__(self, bert_model, output_dim, layer_pulled=-1, add_linear=None, alignment=None, kge_emb_dim=0):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        self.bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.alignment = alignment
        if alignment:
            self.alignment_layer = nn.Linear(output_dim, 768)

        self.bert_model = bert_model

        self.add_linear = add_linear
        # self.kge_emb_dim = kge_emb_dim + 1 if kge_emb_dim > 0 else kge_emb_dim

        self.kge_emb_dim = kge_emb_dim
        self.output_dim = output_dim

    def forward(self, token_ids, attention_mask, segment_ids, kge_embeds=None):
        output_bert, output_pooler = self.bert_model(
            token_ids, attention_mask=attention_mask, token_type_ids=segment_ids
        )

        embeddings = output_bert[:, 0, :]

        return embeddings

        # get embedding of [CLS] token
        # if self.additional_linear is not None:
        #     import torch
        #     if kge_embeds is not None:
        #         embeddings = torch.cat((output_pooler, kge_embeds), 1)
        #     else:
        #         embeddings = output_pooler
        #
        # else:
        #     embeddings = output_bert[:, 0, :]
        #
        # # in case of dimensionality reduction
        # if self.additional_linear is not None:
        #     result = self.additional_linear(self.dropout(embeddings))
        # else:
        #     result = embeddings
        #
        # return result

    def add_layer(self):
        if self.add_linear:
            self.additional_linear = nn.Linear(self.kge_emb_dim+self.bert_output_dim, self.output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def pass_through_last_layer(self, kg_emb, output):
        result = None

        if self.additional_linear is not None:
            import torch
            if kg_emb is not None:
#                log(f"{torch.cat((output, kg_emb), dim=2).size()}", 2)
                embeddings = torch.cat((output, kg_emb), dim=2)

#                log(f"addition layers inputsize = {self.kge_emb_dim}+{self.bert_output_dim}", 2)

                result = self.additional_linear(self.dropout(embeddings))

        return result
