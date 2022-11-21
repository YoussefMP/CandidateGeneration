from Code.FileIOManager import ModelFileIO
from Code.Logger import log
import torch
import json
import os

from Code.blink.biencoder import BiEncoderRanker


def load_parameters(file_manager, model_name, idx_embedding_size):
    if os.path.exists(f"{file_manager.source_folder}/Models/BLINK_output/{model_name}.json"):
        models_path = os.path.join(file_manager.source_folder, "Models/BLINK_output")
        config_name = model_name.replace(".bin", ".json")
        log(f"Loading fine_tuned models config from {models_path}...", 1)
    else:
        models_path = os.path.join(file_manager.source_folder, "Models/")
        config_name = "biencoder_wiki_large.json"
        log(f"loading the default config_file from {file_manager.source_folder}/Models/", 1)

    with open(f"{models_path}{config_name}") as j_file:
        params = json.load(j_file)
        params["kge_emb_dim"] = idx_embedding_size

    return params


def get_models_path(file_man, model_name):

    models_folder = os.path.join(file_man.source_folder, "Models")

    log(f"Looking in {models_folder}: => {os.listdir(models_folder)}", 2)
    log(f"Looking in {os.path.join(models_folder, 'BLINK_output')}: =>"
        f" {os.listdir(os.path.join(models_folder, 'BLINK_output'))}", 2)

    if model_name in os.listdir(models_folder):
        return os.path.join(models_folder), False
    elif model_name in os.listdir(os.path.join(models_folder, "BLINK_output")):
        return os.path.join(models_folder, "BLINK_output"), False

    return os.path.join(file_man.source_folder, "Models/biencoder_wiki_large.bin"), True


def load_model(args):
    model_name = args["modelname"]
    data_folder = args["datafolder"]
    index_name = args["indexname"]

    log("Instantiating the FileIO instance", 0)
    file_manager = ModelFileIO(data_folder, index_name)

    if args.get("debug"):
        return None, file_manager

    log("Reading config file of the pretrained model", 0)
    params = load_parameters(file_manager, model_name, args["index_embedding_size"])
    reranker = BiEncoderRanker(params)

    log("Loading the model's weights...", 0)
    models_path, default = get_models_path(file_manager, model_name)

    if default:
        reranker.load_model(models_path)
        reranker.add_layer()
    else:
        reranker.add_layer()
        reranker.load_model(models_path)

    log("Successfully loaded models weights...", 1)

    return reranker, file_manager
