# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


class BertEncoder(nn.Module):
    def __init__(self, bert_model, output_dim, layer_pulled=-1, add_linear=None, alignment=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.alignment = alignment
        if alignment:
            self.alignment_layer = nn.Linear(output_dim, 768)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, attention_mask, segment_ids, inputs_embeds=None):
        if token_ids is not None:
            outputs = self.bert_model(
                token_ids, attention_mask=attention_mask, token_type_ids=segment_ids
            )
        elif inputs_embeds is not None:
            if self.alignment:
                inputs_embeds = self.alignment_layer(inputs_embeds)
            outputs = self.bert_model(attention_mask=attention_mask, token_type_ids=segment_ids, inputs_embeds=inputs_embeds)
        else:
            raise TypeError

        output_pooler = outputs["pooler_output"]
        output_bert = outputs["last_hidden_state"]

        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result

