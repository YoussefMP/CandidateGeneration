import os
import os.path
import sys

sys.path.insert(0, os.getcwd())

from torch.utils.data import DataLoader
from Code.Logger import log
from Code.blink.biencoder import BiEncoderModule, BiEncoderRanker
from Code.FileIOManager import ModelFileIO
from Code.Data_Manager import EmbeddingsMyDataset
import torch


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


def add_training_params(original, path, batch_size, lr, epochs, index_emb_size):

    params = {
        "kge_emb_dim": index_emb_size,
        "output_path": f"{path}/Models/BLINK_output",
        "encode_batch_size": batch_size,
        "learning_rate": lr,
        "num_train_epochs": epochs,
    }
    overwrite = ["learning_rate", "num_train_epochs", "", ""]

    for k, v in params.items():
        if k in original.keys() and k not in overwrite:
            print(f"{k} is already present")
        else:
            original[k] = v

    return original


def load_params(file_manager, model_name, arg=None):
    log("Reading config file of the pretrained model", 0)
    import json

    if model_name == "":
        model_name= "biencoder_wiki_large.bin"

    config_name = model_name.replace(".bin", ".json")
    models_path = file_manager.models_repo
    with open(f"{models_path}/{config_name}") as j_file:
        params = json.load(j_file)

    if arg and type(arg) == dict:
        log("Instantiating the bi_encoder with ddict args...", 0)
        if arg is None:
            batchsize = 8
            idx_emb_size = 100
        else:
            batchsize = int(arg["batchsize"])
            idx_emb_size = int(arg["index_emb_size"])

        params = add_training_params(params, file_manager.source_folder, 0, 0,
                                      batchsize, idx_emb_size)
    elif type(arg) != dict:
        log("Instantiating the bi_encoder...", 0)
        params = add_training_params(params, file_manager.source_folder,
                                     int(arg.batchsize), float(arg.lr), int(arg.epochs), int(arg.index_emb_size))
    return params, models_path


def load_model(params, models_path, model_name):
    reranker = BiEncoderRanker(params)
    log("Loading the model's weights...", 0)
    #reranker.load_model(f"{models_path}/{model_name}")
    reranker.load_model(f"{models_path}/BLINK_output/epoch_6/pytorch_model.bin")
#    state_dict = torch.load(f"{models_path}{model_name}")
#    reranker.model.load_state_dict(state_dict, strict=False)
    reranker.add_layer()

    return reranker


def main(arg):

    log("Reading the arguments...", 0)
    model_name = arg.modelname

    log("Instantiating the FileIO instance", 0)
    file_manager = ModelFileIO()

    # Reading the json file and adding the parameters for training
    params, models_path = load_params(file_manager, model_name, arg)

    # Instantiating the bi-encoder model for training
    log("Loading the models weights.", 0)
    reranker = load_model(params, models_path, model_name)

    # Calling the training function
    from Code.blink import train_biencoder
    time_taken = train_biencoder.main(reranker, params, file_manager)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--modelname",
                        help="the name of the file in which the model will be saved after training", required=True)
    parser.add_argument("--index_emb_size",
                        help="The size of the embeddings vectors of the entities in the KGE", required=True)

    # Training's hyperparameter
    parser.add_argument("--batchsize", required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--lr", help="learning rate", required=True)

    # Candidates set's size
    parser.add_argument("--setsize", help="Size of the set of candidates to be generated", required=True)
    args = parser.parse_args()

    main(args)
