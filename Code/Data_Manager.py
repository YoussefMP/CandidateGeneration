import torch


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
    def __init__(self, mentions, entities):
        self.x = []
        self.y = []

        idx = []
        for tid in range(len(mentions)):
            if len(list(mentions)[tid]) == 0:
                idx.append(tid)

        for entry in range(len(mentions)):
            for entry_id in range(len(list(mentions)[entry])):
                if list(entities)[entry][entry_id] is not None:
                    self.x.append(torch.IntTensor(list(mentions)[entry][entry_id]))
                    self.y.append(list(entities)[entry][entry_id])

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
