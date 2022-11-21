# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
import random
import time
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.modeling_utils import WEIGHTS_NAME
from Code.blink.biencoder import BiEncoderRanker, load_biencoder
import Code.blink.candidate_ranking.utils as utils
from Code.blink.common.optimizer import get_bert_optimizer
from Code.blink.common.params import BlinkParser
from Code.Logger import log
import faiss
import pickle
import json
import requests
from Code.blink.indexer.faiss_indexer import DenseHNSWFlatIndexer

logger = None


# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(reranker, eval_dataloader, params, device, logger):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        # print(f"{type(batch[0])} == > {batch[0].size()}")
        # print(f"{batch[1][0].size()}=> {batch[1][0].dtype} & {batch[1][1].size()} => {batch[1][1].dtype}")

        batch = [batch[0].to(device), batch[1][0].to(device), batch[1][1].to(device)]
#        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_emb, candidate_input = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input, kge_emb=candidate_emb)

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(
                torch.arange(params["eval_batch_size"])
        ).numpy()
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    log("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    log(" Num optimization steps = %d" % num_train_steps)
    log(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def get_candidates(faiss_idx, e_mapping, ctxt_emb, kge_idx, g_ent):

    # Query Faiss index to get predicted candidates set
    D, candidates = faiss_idx.search_knn(ctxt_emb.cpu().detach(), 10)

    candidates_urls = []
    candidates_ctxt_emb = []
    candidates_kge = []
    labels = []

    for batch in range(len(candidates)):
        candidates_urls.append([])
        candidates_ctxt_emb.append([])
        candidates_kge.append([])
        labels.append([])
        for cid in candidates[batch]:
            try:
                candidates_urls[-1].append(e_mapping[cid])
                if e_mapping[cid] == g_ent[batch]:
                    labels[-1].append(1)
                else:
                    labels[-1].append(0)

            except KeyError:
                candidates_urls[-1].append(list(e_mapping.keys())[cid])
                if list(e_mapping.keys())[cid] == g_ent[batch]:
                    labels[-1].append(1)
                else:
                    labels[-1].append(0)

            candidates_ctxt_emb[-1].append(faiss_idx.index.reconstruct(int(cid)).tolist())
            candidates_kge[-1].append(kge_idx.reconstruct(int(cid)).tolist())

        candidates_urls[-1].append(g_ent[batch])
        candidates_ctxt_emb[-1].append(faiss_idx.index.reconstruct(e_mapping.index(g_ent[batch])).tolist())
        candidates_kge[-1].append(kge_idx.reconstruct(e_mapping.index(g_ent[batch])).tolist())
        labels[-1].append(1)

    # return KG embeddings                  :
    # return entity_description embeddings  :
    return candidates_kge, candidates_ctxt_emb, torch.FloatTensor(labels)



def resize_candidate_embeddings(vectors):
    vectors = torch.FloatTensor(vectors)
    new_size = list(vectors.size())
    new_size[-1] = new_size[-1]-1

    holder = torch.empty(new_size)
    for i in range(new_size[0]):
      for j in range(new_size[1]):
        val = vectors[i][j][-1]
        holder[i][j] = vectors[i][j][vectors[i][j]!=val]

    return holder




def compute_labels(negative_cands, golden_cand):
    labels = []


    batch = torch.cat((golden_cand.unsqueeze(1), negative_cands), dim=1)

    for i in range(golden_cand.size(0)):
        labels.append([])
        for j in batch[i]:
            if golden_cand[i].equal(j):
                labels[-1].append(1)
            else:
                labels[-1].append(0)

    labels = torch.FloatTensor(labels)

    return labels, batch

def get_cand(idx, qv):
    D, candidates = idx.search_knn(qv.cpu().detach(), 10)

    candidates_urls = []
    candidates_ctxt_emb = []

    for batch in range(len(candidates)):
        candidates_urls.append([])
        candidates_ctxt_emb.append([])

        for cid in candidates[batch]:
            candidates_ctxt_emb[-1].append(idx.index.reconstruct(int(cid)).tolist()[:-1])

    return candidates_ctxt_emb

def main(reranker, params, file_manager):


    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # Init model
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved
    # by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
#    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    log("Preparing the data for training ... ", 0)
    from Dataset_processing import get_mentions_entities_dataset
    mentions_context, entities = get_mentions_entities_dataset(file_manager, "aida_train", "training",
                                                               reranker.tokenizer, testing=None)

    from Code.Data_Manager import EmbeddingsMyDataset, SimpleDataset
    log(f"Setting the dataloader with the training data...{len(list(mentions_context.values()))}", 0)
    train_dataloader = EmbeddingsMyDataset(list(mentions_context.values()), list(entities.values()), tokenizer)
    log(f"Loaded pairs and entities => size of the dataset {len(train_dataloader)}", 1)

    log(f"Loading the mapping of ids to urls...", 0)
    entity_mapping = pickle.load(open(f"{file_manager.results_repo}/init_blink_url_map_20_11_22.pkl", "rb"))

#    dataloader = SimpleDataset(mentions_context, entities, tokenizer, entity_mapping)
    train_dataloader = DataLoader(dataset=train_dataloader, batch_size=int(params["train_batch_size"]),
                                  shuffle=True, num_workers=2)

    # TODO: Build Faiss index and needed dictionary
    log(f" Loading the faiss index from the file for the blink embeddings...", 0)
    index = DenseHNSWFlatIndexer(params["out_dim"])
    index.index = faiss.read_index(f"{file_manager.results_repo}/init_blink_emb_20_11_22")
    f_idx = index.index

    # KGE
    log(f"Loading the transE embedding into the index ...", 0)
    kge_index = DenseHNSWFlatIndexer(100)
    kge_index.index = faiss.read_index(f"{file_manager.results_repo}/TransE_index_20_11")
    k_idx = kge_index.index

    new_idx = DenseHNSWFlatIndexer(1024)
    new_idx.deserialize_from(f"{file_manager.results_repo}/final_trained_blink_emb_21_11")

    time_start = time.time()
    log("Starting training...", 0)
    log("device: {}, n_gpu: {}, distributed training: {}".format(device, n_gpu, False))

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_dataloader))
    model.train()

    best_epoch_idx = -1
    best_score = -1
    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        log(f"len iter {len(iter_)} ==== len dataloader {len(train_dataloader)}", 4)
        log(f"DEVICE ====> {device}")

        for step, batch in enumerate(iter_):
            batch = (batch[0].to(device), batch[1][1].to(device), batch[1][0].to(device))
            # batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, kge_input = batch

            loss, _ = reranker(context_input, candidate_input, kge_emb=kge_input)

#            embeddings = reranker.encode_context(context_input)
#            cand_vecs = get_cand(new_idx, embeddings)
#            loss = reranker.score_candidates(embeddings.unsqueeze(1), cand_encs=torch.FloatTensor(cand_vecs).permute(0,2,1).to(device))

#            context_input, entity = batch

#            log(f"Compute mentions context embedding for vector of shape {context_input.numpy().shape}", 1

#            m_ctxt_emb = reranker.encode_context(context_input.to(device))
#            log(f"Ouput of size {m_ctxt_emb.size()}", 2)

#            import pdb
#            pdb.set_trace()

#            log("Use embedding to retrieve negatives (if model's prediction is not accurate)", 1)
#            candidates_kge, candidates_emb, labels = get_candidates(index, entity_mapping, m_ctxt_emb, k_idx, entity)
#            m_ctxt_emb.to(device)
#            log(f"Returned candidates_kge of [8 x {{X : [list : 100]}}] and returned candidates_embeddings in form of {np.array(candidates_emb).shape}", 2)

#            candidates_emb = resize_candidate_embeddings(candidates_emb)
#            candidates_kge = resize_candidate_embeddings(candidates_kge)
#            log(f"________________________________________________________{candidates_emb.size()} ", 2)

#            log("Compute the new embeddings for the negatives (Including the description and relational information about the entity)", 1)
#            candidates_output = reranker.compute_new_embeddings(candidates_emb, candidates_kge)
#            log(f"models new embeddings ==> Candidates_output {candidates_output.size()} ", 2)

#            log(f"Compute the embeddings for the golden entity with candidate_input({candidate_input.dtype}) = {candidate_input.shape} and kge_input({kge_input.dtype}) = {kge_input.size()}", 1)
#            golden_entity_emb = reranker.encode_candidate(candidate_input, kge_input)
#            labels, full_batch = compute_labels(candidates_output.to(device), golden_entity_emb.to(device))

#            loss, _ = reranker.compute_loss(m_ctxt_emb, candidates_output, labels)
            # loss, _ = reranker(context_input, candidates_output)

            if n_gpu > 1:
                loss = loss.mean()                                  # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps + 50) == 0:
                log("Step {} - epoch {} average loss: {}\n".format(
                    step, epoch_idx, tr_loss / (params["print_interval"] * grad_acc_steps),)
                )
                tr_loss = 0

            loss.backward()
            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
            #     log("Evaluation on the development dataset")
            #     evaluate(
            #         reranker, valid_dataloader, params, device=device, logger=logger,
            #     )
            #     model.train()
            #     log("\n")

        log("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        print(f"Saving the model to ---------> {epoch_output_folder_path}")
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        if epoch_idx == 0:
          # Load Validation data
          mentions_context, entities = get_mentions_entities_dataset(file_manager, "aida_testa", "validation",
                                                                     reranker.tokenizer)

          dataloader = EmbeddingsMyDataset(list(mentions_context.values()), list(entities.values()), tokenizer)
          valid_dataloader = DataLoader(dataset=dataloader, batch_size=int(params["eval_batch_size"]),
                                        shuffle=True, num_workers=2)

        print(f"Eval file will be saved in {epoch_output_folder_path}")
        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        log("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    log("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    log("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    print(f"Path to model is set to {params['path_to_model']}")
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
