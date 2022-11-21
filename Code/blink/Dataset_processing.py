from Code.Logger import log
import torch
import re
import csv
import os

__DEBUG__ = False
DEBUG_SIZE = None

ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"


def str_to_tensor(t_str):
    if "tensor" in t_str:
        t_str = t_str[7:].replace(",", "")
    emb = t_str.strip("[]()").split()
    tensor = []
    for val in emb:
        try:
            tensor.append(float(val))
        except ValueError:
            pass
    return tensor


def extract_mentions_context(text):

    s_tag = ENT_START_TAG
    e_tag = ENT_END_TAG

    contexts = []

    text_list = text.split()
    in_mention = False

    mention = ""
    p_context = ""
    s_context = ""

    for wid in range(len(text_list)):

        if in_mention:
            if text_list[wid] != e_tag and e_tag not in text_list[wid]:
                mention += f" {text_list[wid]}"
            else:
                in_mention = False
                for i in range(len(text_list) - wid):
                    if text_list[i+wid] != s_tag and text_list[i+wid] != e_tag:
                        if s_tag in text_list[i+wid]:
                            addition = text_list[i+wid].replace(s_tag, "")
                        elif e_tag in text_list[i+wid]:
                            addition = text_list[i+wid].replace(e_tag, "")
                        else:
                            addition = text_list[i+wid]
                        s_context += f"{addition} "

                    if len(s_context.split()) == 32:
                        break
                contexts.append({"mention":mention, "context_left":p_context, "context_right":s_context})
                p_context = ""
                mention = ""
                s_context = ""

        if text_list[wid] == s_tag or s_tag in text_list[wid]:
            for i in range(wid, -1, -1):
                if text_list[i] != s_tag and text_list[i] != e_tag:
                    if s_tag in text_list[i]:
                        addition = f"{text_list[i].replace(s_tag, '')} " if \
                            len(p_context) > 0 else f"{text_list[i].replace(s_tag, '')}"
                    elif e_tag in text_list[i]:
                        addition = f"{text_list[i].replace(e_tag, '')} " if \
                            len(p_context) > 0 else f"{text_list[i].replace(e_tag, '')}"
                    else:
                        addition = f"{text_list[i]} " if len(p_context) > 0 else f"{text_list[i]}"

                    p_context = addition + p_context
                if len(p_context.split()) == 32:
                    break
            in_mention = True
    return contexts


def read_complete_entity_dataset(path):

    file = open(path, "r", encoding="utf-8")
    reader = csv.reader(file, delimiter="\t")

    entity_dataset = {}
    for row in reader:
        if row:
            entity_dataset[row[0]] = (str_to_tensor(row[1]), row[2], row[3])
    file.close()

    return entity_dataset


def write_complete_entity_dataset(file_manager, labels, abstracts):
    """
    Write tsv file: Row Format = URL \t Embedding \t Label \t Description
    :param labels:
    :param abstracts:
    :return:
    """
    file_path = f"{file_manager.repo}/complete_entity_dataset.tsv"

    file = open(file_path, "w", encoding='utf8')
    writer = csv.writer(file, delimiter="\t")

    entity_dataset = {}
    for entity in labels.keys():
        try:
            description = abstracts[entity]
            writer.writerow([entity, labels[entity][1], labels[entity][0], description])
            entity_dataset[entity] = (labels[entity][1], labels[entity][0], description)
        except KeyError:
            writer.writerow([entity, labels[entity][1], labels[entity][0], "NaN"])
            entity_dataset[entity] = (labels[entity][1], labels[entity][0], "NaN")
    file.close()

    return entity_dataset


def extend_entity_dataset(original, additional_data, tokenizer):

    from Code.blink.data_process import get_candidate_representation

    n_count = 0
    t_count = 0
    max_length = 0
    for key in original:
        for v_idx in range(len(original[key])):
            t_count += 1
            try:
                data_item = additional_data[original[key][v_idx]]
                tokenized_ent_desc = get_candidate_representation(data_item[2], tokenizer, 196, data_item[1])["ids"]

#                tokenized_ent_desc = tokenizer.tokenize(f"[CLS]{data_item[1]}[ENT]{data_item[2]}[SEP]")
#                tokenized_ent_desc = tokenizer.convert_tokens_to_ids(tokenized_ent_desc)
#                tokenized_ent_desc += [tokenizer.convert_tokens_to_ids('[PAD]')] * (196 - len(tokenized_ent_desc))
#                if len(tokenized_ent_desc) > max_length:
#                    max_length = len(tokenized_ent_desc)

                original[key][v_idx] = (torch.FloatTensor(data_item[0]), torch.IntTensor(tokenized_ent_desc))

            except TypeError:
                t_count -= 1
                original[key][v_idx] = None
            except KeyError:
                n_count += 1
                original[key][v_idx] = None

    log(f"From a total of {t_count} we lost {n_count} entries", 3)
    return original


def generate_ent_set(data):
    """
    Generate a set of all entities in the dataset without duplicates
    :param data:
    :return:
    """
    entities_set = {}

    for k, v in data.items():
        for entry in v:
            if entry:
                entities_set[entry[0]] = entry[1]

    return entities_set


def filter_entities_for_description(file_manager, data_entities, criteria=None):
    """
    Read the DBpedia dump file and extract labels and descriptions of the entities present in the dataset
    :param data_entities:
    :param criteria
    :return: complete_entities_dataset {txt_id: [{link: str: DBpedia_link,
                                                 label: str: entity label,
                                                 abstract: str: entity description
                                                }]
                                        }
    """

    file_path = f"{file_manager.repo}/DBpedia dumps/"

    if criteria == "label":
        file_path += "labels_lang=en.ttl"
        regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<label>.*)"@en''')
    else:
        file_path += "short-abstracts_lang=en.ttl"
        regex = re.compile(r'''<(?P<link>.*?)>.*?<.*?> "(?P<abstract>.*)"@en''')

    entities = generate_ent_set(data_entities)

    complete_entities_dataset = {}
    count = 0

    log(f"Filtering entities with no {criteria}", 2)

    with open(
            file_path,
            "r", encoding="utf-8") as file:
        line = "a"
        t = 0
        while line:
            t += 1
            line = file.readline()
            m = regex.match(line)

            if m:

                full_link = m.group("link") in entities.keys()
                partial_link = m.group("link").replace("http://dbpedia.org", "") in entities.keys()

                if full_link or partial_link:
                    count += 1
                    try:
                        key = m.group("link").replace("http://dbpedia.org", "") if partial_link else\
                            m.group("link") if full_link else None

                        if criteria == "label":
                            complete_entities_dataset[m.group("link")] =\
                                (m.group("label"),
                                 torch.FloatTensor(entities[key]))
                        else:
                            complete_entities_dataset[key] = m.group("abstract")

                    except KeyError as k_err:
                        print(f"KeyError: {k_err} \n ")

            if t % 1500000 == 0:
                log(f"Parsed = {t} lines, and found data for {count}/{len(entities.keys())} entities", 4)

            if len(entities.keys()) == count and criteria == "label":
                break
        log(f"Results of filtering entities with no {criteria} => {count} / {len(entities)} = {count/len(entities)}%", 3)

    file.close()
    return complete_entities_dataset


def load_ent_emb(file_manager):
    import csv

    ent_emb_index = {}

    data = open(f"{file_manager.results_repo}/sub_kge_18_11_22.tsv", 'r', encoding="utf-8")
    reader = csv.reader(data, delimiter='\t')

    log(f"Reading the sub_kge file ")

    for row in reader:
        if len(row) > 0:
            #ent_emb_index[row[0]] = str_to_tensor(row[1])
            ent_emb_index[row[0]] = torch.FloatTensor([float(i) for i in row[1:]])

    return ent_emb_index


def filter_entities(file_manager, ordered_golden_entities):
    """
    Remove entities from the dataset for which no embeddings in the index exists
    :param ordered_golden_entities:
    :return: ordered_entity_embedding {
                                      txt_id: [
                                               (str: url, list: embedding),
                                              ]
                                    }
    """
    # Load the entity embeddings in a dictionary and extract the one we found in our data
    entity_embeddings_index = load_ent_emb(file_manager)
    ordered_entity_embeddings = {}

    p_count = 0
    n_count = 0

    for k, v in ordered_golden_entities.items():
        if not v:
            ordered_entity_embeddings[k] = []
        for entity in v:
            if ordered_entity_embeddings.get(k):
                try:
                    ordered_entity_embeddings[k].append((entity, entity_embeddings_index[entity]))
                    p_count += 1
                except KeyError:
                    ordered_entity_embeddings[k].append(None)
                    n_count += 1
            else:
                try:
                    ordered_entity_embeddings[k] = [(entity, entity_embeddings_index[entity])]
                    p_count += 1
                except:
                    n_count += 1
                    ordered_entity_embeddings[k] = [None]

    log(f"From the set of all entities found embeddings for {p_count} and lost {n_count}", 2)
    return ordered_entity_embeddings


def prepare_data(file_manager, file, file_type):
    """
    Read the file and extract mentions with their corresponding golden entity
    :param file_manager:    ModelFileIO
    :param file:            file_name
    :param file_type:       key under which the file path is saved
    :return: texts                  : {txt_id: str: annotated_text        }  # mentions tagged [m] [/m]
             ordered_golden_entities: {txt_id: [str: DBpedia_links]       }
             ordered_mentions       : {txt_id: [str: mention_surface_form]}
    """
    log("Read document and extract Ordered datasets...", 2)
    debug_size = DEBUG_SIZE if __DEBUG__ else None
    texts, ordered_golden_entities, ordered_mentions = \
        file_manager.read_aida_input_doc(file_type, debug_size=debug_size, file_name=file)

    lens_eg = [len(ordered_golden_entities[i]) for i in range(len(ordered_golden_entities))]
    lens_mc = [len(ordered_mentions[i]) for i in range(len(ordered_mentions))]

    return texts, ordered_golden_entities, ordered_mentions


def get_mentions_entities_dataset(file_manager, file, file_type, tokenizer, testing=None):
    """
    Prepares the entity, mention embeddings pairs for the training loop.
    :param file: file name, from which the data will be read and loaded
    :param file_type: types differentiate between training and validation
    :param tokenizer: Tokenizer of the model
    :param file_manager
    :return:
    """

    log("Getting the data from the data files...", 1)
    # Get annotated texts and the list of mentions and entities in the file
    texts, ordered_golden_entities, mentions = prepare_data(file_manager, file, file_type)

    if testing is None:
#      log("Filtering the entities with no embeddings in the KGE...", 1)
#      ordered_entity_embeddings = filter_entities(file_manager, ordered_golden_entities)

      if os.path.exists(f"{file_manager.results_repo}/Complete_17_11_22.tsv"):
          # TODO: Remove eval_ form file name
          log("Reading directlz from the file ...", 1)
          entity_data = read_complete_entity_dataset(f"{file_manager.results_repo}/Complete_17_11_22.tsv")
      else:
          log("Filtering entities with no labels and descriptions available in the Wiki dump", 1)
          # # Get the entities for which we have a label and a description
          entity_label_dict = filter_entities_for_description(file_manager, ordered_entity_embeddings, criteria="label")
          entity_abstract_dict = filter_entities_for_description(file_manager, ordered_entity_embeddings, criteria="abstract")
          entities_data = write_complete_entity_dataset(file_manager, entity_label_dict, entity_abstract_dict)

      log(f"Extending the entity dataset with [CLS]label[ENT]description[SEP] labels and abstracts...", 1)
      ordered_entity_embeddings = extend_entity_dataset(ordered_golden_entities, entity_data, tokenizer)

      # Get mentions contexts maximal length is 78 => 32 word pieces + mention + 32 word pieces
    ordered_mention_contexts = {}

    log(f"Extracting mentions contexts (window size = 64) and saving them in tokens form...", 1)
    log(f"Output form => ordered_mention_contexts = {{tid: [[int:token_id ], ]}}", 1)
    # for text in range(len(texts)):

    for text in range(len(texts)):
        # tokenized_text = tokenizer.tokenize(texts[text])
        # text_token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        ordered_mention_contexts[text] = extract_mentions_context(texts[text])

    if testing is None:
        return ordered_mention_contexts, ordered_entity_embeddings
    else:
        return ordered_mention_contexts, ordered_golden_entities
