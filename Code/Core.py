import configparser

import torch
from torch.utils.data import DataLoader
import argparse

from transformers import WEIGHTS_NAME

from Gen_Entities_Embeddings_Index import gen_ent_emb_index_from_list, write_results_file
from Logger import log
from Data_Manager import *
from MentionsEncoder import Encoder
from FileIOManager import ModelFileIO
from Train_Encoder import train_model
from Attention_Encoder import initiate_model
from Evaluate_Model import eval_model, eval_biencoder_model
import os
from blink import blink_Core


# Model's Parameters
MODEL_PARAMS = {"source_size": 384,
                "target_size": 25,
                "nb_layers": 12,
                "nb_heads": 6,
                "fwd_expansion": 4,
                "dp": 0.1
                }

# Loss function // this feature does not work youll have to change the loss function manually
# TODO: Implement
LOSS_FUNC = "mse+cos"


def time_format(seconds: int):
    if seconds is not None:
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        if d > 0:
            return '{:02d}D {:02d}H {:02d}m {:02d}s'.format(d, h, m, s)
        elif h > 0:
            return '{:02d}H {:02d}m {:02d}s'.format(h, m, s)
        elif m > 0:
            return '{:02d}m {:02d}s'.format(m, s)
        elif s > 0:
            return '{:02d}s'.format(s)
    return '-'


# #################################################################################################
def results_file_ready(path, file):
    # Path under which the files will be saved or from where the files will be loaded
    me_set = path + "/Results/" + file

    log(f"Searching for {me_set}", 1)

    if os.path.exists(me_set):
        log("Results files are ready", 2)
        return True

    log("Results files could not be found", 2)
    return False


def write_ini_file(model_name, evaluation_results):
    model_config_name = f"{model_name}_config.ini"
    model_config_path = f"./../Models/{model_config_name}"

    config_writer = configparser.ConfigParser()
    config_writer["Evaluation results"] = {}
    # for k, v in model_params:
    #     config_writer["Model parameters"] = {k: v}

    for k, v in evaluation_results.items():
        config_writer["Evaluation results"][k] = str(v)

    with open(model_config_path, "a", encoding="utf-8") as out_file:
        config_writer.write(out_file)
    out_file.close()


# #################################################################################################
def read_model_params(path):
    print(path, os.path.exists(path))
    config = configparser.ConfigParser()
    config.read(path)
    model_params = {}
    for key in config["Model_Parameters"]:
        if key == "dp":
            model_params[key] = float(config["Model_Parameters"][key])
        else:
            model_params[key] = int(config["Model_Parameters"][key])

    print(model_params)
    return model_params


# #################################################################################################
def input_loop(key):
    strings_dict = {
        "source_size": "Enter input size of model (must divide 768): ",
        "target_size": "Enter size of the output vector (must equal the embedding dimension of the KGE): ",
        "nb_layers": "Enter number of layers: ",
        "nb_heads": "Enter number of attention heads (must divide the source_size default=12): ",
        "fwd_expansion": "forward expansion: ",
        "dp": "dropout: "
    }

    convertible = False
    while not convertible:
        value = input(strings_dict[key])
        try:
            if key == "dp":
                value = float(value)
            else:
                value = int(value)
            convertible = True
        except ValueError:
            if value == "":
                convertible = True
            else:
                print("please enter a valid value or press Enter")

    return value


def ask_for_params(parms):
    results = {}
    for key in parms.keys():
        results[key] = input_loop(key)
    return results


def init_config(model_name, model_params, training_params):
    model_config_name = f"{model_name}_config.ini"
    model_config_path = f"./../Models/{model_config_name}"

    config_writer = configparser.ConfigParser()
    config_writer["Model_Parameters"] = {}
    for k, v in model_params.items():
        config_writer["Model_Parameters"][k] = str(v)
    config_writer["Training parameters"] = {}
    for k, v in training_params.items():
        config_writer["Training parameters"][k] = str(v)

    with open(model_config_path, "w", encoding="utf-8") as out_file:
        config_writer.write(out_file)
    out_file.close()


# #################################################################################################
def prepare_dataset(file_manager, file_type):

    dataset = ModelFileIO.read_data_pairs_file(file_manager.files[file_type])
    # dataset = file_manager.load_specific_data(file_manager.files[file_type], 50)

    dataloader = NewEmbeddingsMyDataset(dataset)

    return dataloader


def prepare_data(file_manager, data_files, file_types, data_path, index_name):

    instantiated = False

    for file in data_files:
        i = data_files.index(file)

        # results file name
        pairs_file_name = f"{index_name[:index_name.find('_')+1]}mention_entity_pairs_" + file + ".tsv"
        txt_files_name = f"annotated_text_chunks_" + file + ".txt"

        if not results_file_ready(data_path, pairs_file_name):
            if not instantiated:
                log("Instantiating model for encoding mentions...", 0)
                encoder = Encoder()
                instantiated = True

            log(f"Reading {file_types[i]} data file ...", 1)
            file_manager.add_path(file_types[i], file, "Datasets")
            texts, ordered_golden_entities, ordered_mentions = file_manager.read_aida_input_doc(file_types[i])

            ent_emb_file_path = f"/Results/{index_name[:index_name.find('_')]}_entity_embeddings.tsv"
            if not os.path.exists(data_path + ent_emb_file_path):
                log("Creating the index of golden entities embeddings...", 1)
                eg_index = gen_ent_emb_index_from_list(ordered_golden_entities, index_name)
                write_results_file(eg_index, data_path + ent_emb_file_path)

            if not os.path.exists(data_path + f"/Results/mention_entity_pairs_{file}.tsv"):
                log("Embedding tagged mentions in the texts...", 1)
                paired_data, corpus_chunks = encoder.embed_mentions(texts, ordered_golden_entities,
                                                                    ordered_mentions)
            else:
                corpus_chunks = None
                paired_data = file_manager.read_data_pairs_file(data_path + f"/Results/mention_entity_pairs_{file}.tsv")

            log("Writing mention-entity pairs file and the text chunks file ...", 0)
            file_manager.add_path("res_" + file_types[i], pairs_file_name, "Results")
            file_manager.write_results(paired_data, corpus_chunks,
                                       pairs_file_name, txt_files_name
                                       )
        else:
            file_manager.add_path("res_" + file_types[i], pairs_file_name, "Results")
# #################################################################################################


def process_arguments(arg):

    model_name = arg.modelname
    index_name = arg.indexname
    data_folder_path = arg.datafolder

    training_params = {
        "batch_size": int(arg.batchsize),
        "epochs": int(arg.epochs),
        "learning_rate": float(args.lr),
        "loss_function": LOSS_FUNC
    }

    batch_size = int(arg.batchsize)
    epochs = int(arg.epochs)
    learning_rate = float(arg.lr)

    log("Instantiating the file manager...", 0)
    file_manager = ModelFileIO(data_folder_path, index_name)

    data_files = ["aida_train", "aida_testa", "aida_testb"]
    file_types = ["training", "validation", "evaluation"]

    if "blink" not in model_name:
        log("Reading input files and computing embeddings for pairs", 0)
        prepare_data(file_manager, data_files, file_types, data_folder_path, index_name)

        train_dataset = prepare_dataset(file_manager, "res_training")
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    log("Instantiating model ...", 0)
    if not os.path.exists(f"./../Models/{model_name}") and "blink" not in model_name:
        # If model weights are not available a new model will be instantiated
        if not os.path.exists(f"./../Models/"):
            os.makedirs("./../Models/")
        d_res = ask_for_params(MODEL_PARAMS)
        source_size, target_size, nb_layers, nb_heads, fwd_expansion, dp = d_res.values()

        print(f"instantiating model with")
        model_params = {"source_size": source_size if source_size else MODEL_PARAMS["source_size"],
                        "target_size": target_size if target_size else MODEL_PARAMS["target_size"],
                        "nb_layers": nb_layers if nb_layers else MODEL_PARAMS["nb_layers"],
                        "nb_heads": nb_heads if nb_heads else MODEL_PARAMS["nb_heads"],
                        "fwd_expansion": fwd_expansion if fwd_expansion else MODEL_PARAMS["fwd_expansion"],
                        "dp": dp if dp else MODEL_PARAMS["dp"]}

        print(model_params)
        model = initiate_model(**model_params)
        time_taken = train_model(model, train_dataloader, epochs, learning_rate)
        training_params["time_taken"] = time_format(time_taken)
        init_config(model_name, model_params, training_params)
        torch.save(model.state_dict(), f"./../Models/{model_name}")

    elif not os.path.exists(f"./../Models/Bi_{model_name}/{WEIGHTS_NAME}") and "blink" in model_name:
        model = blink_Core.main(file_manager)
        init_config(model_name, {}, training_params)

        mentions_context, entities = model.get_mentions_entities_dataset("aida_train", "training")

        dataloader = EmbeddingsMyDataset(mentions_context.values(), entities.values())
        train_dataloader = DataLoader(dataset=dataloader, batch_size=batch_size, shuffle=True, num_workers=2)

        time_taken = model.training_loop(train_dataloader, epochs, learning_rate)
        print(f"Time taken for training ==> {time_format(time_taken)}")
        model.save_model_state(model.bi_encoder, f"./../Models/Bi_{model_name}/", model_name)

    elif "blink" not in model_name:
        params = read_model_params(f"./../Models/{model_name}_config.ini")
        model = initiate_model(**params)
        model.load_state_dict(torch.load(f"./../Models/{model_name}"))

    if "blink" in model_name:

        model = blink_Core.main(file_manager)

        state_dict = torch.load(f"./../Models/Bi_{model_name}/{WEIGHTS_NAME}")
        model.bi_encoder.load_state_dict(state_dict)

        model.generate_faiss_entity_index()

        mentions_context, entities = model.get_mentions_entities_dataset("aida_testa", "validation")

        dataloader = EmbeddingsMyDataset(mentions_context.values(), entities.values())
        validation_dataloader = DataLoader(dataset=dataloader, batch_size=batch_size, shuffle=True, num_workers=2)

        evaluation_results = eval_biencoder_model(model, validation_dataloader, index_name)
    else:
        log("Evaluating model...", 0)
        validation_data = prepare_dataset(file_manager, "res_validation")
        validation_dataloader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True, num_workers=2)

        set_size = 10
        evaluation_results = eval_model(model, validation_dataloader, index_name, set_size)

    write_ini_file(model_name, evaluation_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelname",
                        help="the name of the file in which the model will be saved after training", required=True)
    parser.add_argument("--indexname",
                        help="the name of the index from which the entities will be retrieved", required=True)
    parser.add_argument("--datafolder",
                        help="path to the folder containing the data files", required=True)

    # Training's hyperparameter
    parser.add_argument("--batchsize", required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--lr", help="learning rate", required=True)

    # Candidates set's size
    parser.add_argument("--setsize", help="Size of the set of candidates to be generated", required=True)
    args = parser.parse_args()
    process_arguments(args)


