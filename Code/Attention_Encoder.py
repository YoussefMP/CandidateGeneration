import copy
import os.path
import random
import torch
import torch.nn as nn


__DEBUG__ = False


def initiate_model(path, **kwargs):
    parameters = {}

    for k in kwargs:
        parameters[k] = kwargs[k]

    transformer_encoder = TransformerEncoder(**parameters)
    # transformer_encoder = NewTransformer(**parameters)

    print(f"Created the model...", end="")
    if os.path.exists(path):
        print(f"Loaded exisiting model's weights... ")
        transformer_encoder.load_state_dict(torch.load(path))
    else:
        print(f"(Could not find weights for model)")

    return transformer_encoder


def clones(module, n):
    # "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SelfAttention(nn.Module):
    def __init__(self, source_size, nb_heads):
        super(SelfAttention, self).__init__()
        self.input_size = source_size
        self.heads = nb_heads
        self.head_dim = source_size // nb_heads

        # self.head_dim  * heads == embed_size

        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.values = clones(nn.Linear(self.head_dim, self.head_dim), 4)
        # self.keys = clones(nn.Linear(self.head_dim, self.head_dim), 4)
        # self.queries = clones(nn.Linear(self.head_dim, self.head_dim), 4)
        self.linears = clones(nn.Linear(self.head_dim, self.head_dim), 4)
        self.fc_out = nn.Linear(self.heads * self.head_dim, source_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # pprint(f"attention input: values={values.shape}, keys={keys.shape}, query={query.shape}", 4)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, -1, self.heads, self.head_dim)
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        query = query.reshape(N, -1, self.heads, self.head_dim)

        # values = self.values(values)  # (N, value_len, heads, head_dim)
        # keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        # queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        queries, keys, values = \
            [l(x).view(N, -1, self.heads, self.head_dim).transpose(1, 2)
             for l, x in zip(self.linears, (query, keys, values))]

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.input_size ** (1 / 2)), dim=3)
        # attention = torch.sigmoid(energy / (self.input_size ** (1 / 2)))
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, -1, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, source_size, nb_heads, dp, fwd_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(source_size, nb_heads)

        self.norm1 = nn.LayerNorm(source_size)
        self.norm2 = nn.LayerNorm(source_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(source_size, fwd_expansion * source_size),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dp)
        self.feed_forward_out = nn.Linear(fwd_expansion * source_size, source_size)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # pprint(f"attention output = {attention.size()}", 4)

        # pprint(f"input through norm1 layer = {(attention + query).size()}", 4)
        out_norm = self.norm1(attention + query)

        # pprint(f"out_norm => {out_norm.size()}", 4)
        # x = self.dropout(out_norm)

        # pprint(f"dropout_layer output={out_norm.size()}", 4)
        forward = self.feed_forward(out_norm)
        x = self.dropout(forward)
        forward = self.feed_forward_out(x)

        # pprint(f"Forward layer's output={forward.size()}", 4)
        out = self.dropout(self.norm2(forward + out_norm    ))

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, source_size, target_size, nb_layers,
                 nb_heads, fwd_expansion, dp):

        super(TransformerEncoder, self).__init__()
        self.source_size = source_size
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    source_size, nb_heads,
                    dp=dp,
                    fwd_expansion=fwd_expansion
                )
                for _ in range(nb_layers)
            ]
        )

        self.position_embedding = nn.Embedding(source_size, source_size)
        self.dropout = nn.Dropout(dp)

        self.transfer_layer = nn.Linear(source_size, target_size)
        self.pre_trained = False

    def forward(self, source, target=None):

        # x.shape = [1, 768]
        batches, seq_length = source.shape
        positions = torch.arange(0, seq_length).expand(batches, seq_length)  # [batches, seq_length]

        # pprint(f"Positions vector shape= {positions.size()}, source vector shape= {source.shape}", 2)
        dropout_input = source + self.position_embedding(positions).mean(-1)
        # pprint(f"dropout layer input's size= {dropout_input.size()}", 2)

        out = self.dropout(dropout_input)
        # pprint(f"dropout layer output's size= {out.size()}", 2)

        for layer in self.layers:
            # pprint(f"input through the encoder's transformer block => {out.size()}", 3)
            out = layer(out, out, out).mean(1)
            # pprint(f"output of the transformer Nb{l} = {out.shape}", 4)
        # pprint(f"final input => {out.size()}", 2)

        out = self.transfer_layer(out)

        if target is not None:
            scores = out.mm(target.t())
        else:
            scores = None

        return scores, out


class NewTransformer(nn.Module):
    def __init__(self, source_size, target_size, nb_layers,
                 nb_heads, fwd_expansion, dp):

        super(NewTransformer, self).__init__()
        self.source_size = source_size
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    source_size, nb_heads,
                    dp=dp,
                    fwd_expansion=fwd_expansion
                )
                for _ in range(nb_layers)
            ]
        )

        self.position_embedding = nn.Embedding(source_size, source_size)
        self.dropout = nn.Dropout(dp)

        self.transfer_layer = nn.Linear(source_size, target_size)
        # Implement a transformer block instead of Linear layers
        self.joint_layer = nn.Linear(target_size, target_size)

        self.pre_trained = False

    def forward(self, source, target=None):
        # x.shape = [1, 768]
        batches, seq_length = source.shape

        positions = torch.arange(0, seq_length).expand(batches, seq_length)  # [batches, seq_length]
        dropout_input = source + self.position_embedding(positions).mean(-1)

        out = self.dropout(dropout_input)

        for layer in self.layers:
            out = layer(out, out, out).mean(1)

        return out
        # if target:
        #     scores = out.mm(target.t())
        #
        # return scores, out
