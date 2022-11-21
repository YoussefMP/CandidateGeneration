from torch import optim
from Logger import log
import torch.nn as nn
import requests
import torch
import json
import math
import time
import torch.nn.functional as F
import numpy as np


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


def get_entity_embedding(url):

    headers = {
        'Content-Type': 'application/json',
    }
    entity = url.replace("http://dbpedia.org", "")

    json_data = {
        'indexname': 'shallom_dbpedia_index',
        'entities': [
            entity,
        ],
    }

    response = requests.get('http://unikge.cs.upb.de:5001/get-entity-embedding', headers=headers, json=json_data)

    try:
        ans = json.loads(response.text)
        tensor = torch.FloatTensor(ans[entity]).unsqueeze(0)
        return tensor
    except json.decoder.JSONDecodeError:
        print("miss")
        return None


def compute_labels(entities):
    labels = []

    for i in range(len(entities)):
        label = []
        for j in range(len(entities)):
            if entities[i] == entities[j]:
                label.append(1)
            else:
                label.append(0)
        labels.append(label)

    return torch.FloatTensor(labels)


def train_model(encoder, dataset, epochs, learning_rate, rand_neg=True):

    # Training Hyper parameters
    time_sum = 0
    log("Starting the training ...", 1)
    step = 0

    # Loss function and optimizer
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion2 = nn.CosineSimilarity()
    criterion = nn.MSELoss()

    log("Entering Epoch loop", 1)
    # Training Loop
    for epoch in range(epochs):

        e_start = time.time()
        log(f"____ Going through Epoch {epoch} ________ ", 1)

        # set model in training mode
        encoder.train()
        losses = []

        for idx, data in enumerate(dataset):
            # Old
            source = data[0][2]
            target = data[1][1]

            entry_size = int(768 / encoder.source_size)
            source = source.reshape([source.size(0), entry_size, -1])
            source = source.mean(1)

            # Input: [batch_size, inputs_dimensions]
            scores, outputs = encoder(source, target)

            # scores = F.normalize(scores, 2.0, -1)

            optimizer.zero_grad()

            if scores is not None:
                labels = compute_labels(data[1][0])
                loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
                # TODO: add parameters?
                loss = loss_fct(scores, labels)
            else:
                loss = criterion(outputs, target)
                loss_2 = criterion2(outputs, target)
                loss_2 = -loss_2 + 1

                loss = loss + loss_2.mean()

            loss.sum().backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            optimizer.step()

            step += 1

            # losses.append(sum(loss)/len(loss))
            losses.append(loss)

            if (idx % 10) == 0 and idx > 1:
                e_end = time.time()
                sid = 0
                log(f"Epoch_{epoch}: is {(idx * 100) / dataset.__len__()}% Done,"
                    f" achieved loss = {sum(losses) / len(losses)},"
                    f" time for batch process: { e_end - e_start}", 2)

        e_end = time.time()
        log(f"Epoch_{epoch}: Avg-Loss = {sum(losses) / len(losses)}, Epoch lasted: {e_end - e_start}", 0)
        # log(f"Epoch_{epoch}: Avg-Loss = {sum(losses) / len(losses)}, Epoch lasted: {e_end - e_start}", 0)
        time_sum += e_end - e_start

    return time_sum

