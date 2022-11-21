from transformers import BertForSequenceClassification, BertModel
import torch.nn as nn
import torch


class QModel(nn.Module):
    def __init__(self, source_size, target_size):

        self.source_size = 768

        super(QModel, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.ll = nn.Linear(source_size, target_size)

    def forward(self, input_v):

        out = self.model(inputs_embeds=input_v.unsqueeze(0))
        hidden_states = out[2]

        pooled_output = torch.stack(hidden_states, dim=0).permute(1, 2, 0, 3).mean(2).squeeze()
        out = self.ll(pooled_output)

        return None, out
