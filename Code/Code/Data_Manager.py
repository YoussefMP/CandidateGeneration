import torch
from Code.blink.data_process import get_context_representation


class FinalDataset:
    def __init__(self, entries):
        self.x = []
        self.y = []
        self.z = []

        for entry in entries:
            self.x.append(entry[0])
            self.y.append(entry[1])
            self.z.append(entry[2])

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return self.n_samples

class EmbeddingsMyDataset:
    def __init__(self, mentions, entities, tokenizer, testing=None):
        self.x = []
        self.y = []

        idx = []
        for tid in range(len(mentions)):
            if len(list(mentions)[tid]) == 0:
                idx.append(tid)

        for entry in range(len(mentions)):
            for entry_id in range(len(list(mentions)[entry])):
                if list(entities)[entry][entry_id] is not None:
                    representation = get_context_representation(list(mentions)[entry][entry_id], tokenizer, 78)
                    self.x.append(torch.IntTensor(representation["ids"]))

                    if testing is None:
                      self.y.append(
                                  (torch.FloatTensor(list(entities)[entry][entry_id][0]),
                                   list(entities)[entry][entry_id][1])
                                    )
                    else:
                      self.y.append(entities[entry][entry_id])

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class NewEmbeddingsMyDataset:
    def __init__(self, data2):
        self.x = []
        self.y = []

        for d_entry in data2:
            self.x.append((d_entry[0], d_entry[1], d_entry[3]))
            self.y.append((d_entry[2], d_entry[4]))

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class SimpleDataset:

    def __init__(self, data1, data2, tokenizer, ulist):
        self.x = []
        self.y = []

        for i in range(len(data1)):
            for j in range(len(data1[i])):

              if data2[i][j] not in ulist:
                  continue

              if data2[i][j] is not None:
                  representation = get_context_representation(data1[i][j], tokenizer, 78)
                  self.x.append(torch.IntTensor(representation["ids"]))

              self.y.append(data2[i][j])

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

