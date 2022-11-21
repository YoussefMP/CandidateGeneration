import os.path
import re
import csv
import random
import torch
from Code.Logger import log
import time

M_TAG_LENGTH = 9
ENT_START_TAG = "[m]"
ENT_END_TAG = "[/m]"
ENT_TITLE_TAG = "[unused2]"


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


def annotate_text(text, mentions):
    # sort mention in the order they appear in the text
    mentions.sort(key=lambda tup: tup[1])  # sorts in place
    mentions_annotated = 0

    s_tag = ENT_START_TAG
    e_tag = ENT_END_TAG

    for span in mentions:
        # Annotating mention with tags in the text
        start_id = (mentions_annotated * M_TAG_LENGTH) + span[0]
        end_id = (mentions_annotated * M_TAG_LENGTH) + span[1]

        if start_id == 0:
            text = f'{s_tag} ' + text[0: end_id] + f' {e_tag}' + text[end_id:]
        else:
            if text[start_id - 1: end_id].startswith('('):
                text = text[0: start_id] + f' {s_tag}' + text[start_id: end_id] + f' {e_tag}' + text[end_id:]
            else:
                text = text[0: start_id - 1] + f' {s_tag}' + text[start_id - 1: end_id] + f' {e_tag}' + text[end_id:]

        mentions_annotated += 1
    return text


def str_to_tensor(t_str, dim=0):

    if "tensor" in t_str:
        t_str = t_str[7:].replace(",", "")

    emb = t_str.strip("[]()").split()
    tensor = []

    for val in emb:
        try:
            tensor.append(float(val))
        except ValueError:
            pass

    # if len(tensor) != dim and dim != 0:
    #     print(f"smth went wrong. retrieved tensor of size ({len(tensor)})")
    return torch.FloatTensor(tensor)


def str_to_tuple(t_str):

    tuple_elems = t_str.strip("()").split()
    tup = (int(tuple_elems[0].strip(',')), int(tuple_elems[1]))
    return tup


class ModelFileIO:
    def __init__(self, repo_path=None, index_name=None):
        log("Setting the paths and creating folders", 1)
        self.source_folder, self.repo, self.models_repo = self.set_paths()
        self.index_name = index_name
        self.files = {}

        if not os.path.exists(self.repo + "/Results/"):
            os.makedirs(self.repo + "/Results/")

        self.results_repo = self.repo + "/Results/"

    @staticmethod
    def set_paths():
        import os
        import pathlib

        p_path = pathlib.Path(__file__).parent.resolve()
        tries = 0
        while "Models" not in os.listdir(p_path):
            if "CandidatesGenerator" in os.listdir(p_path):
                p_path = os.path.join(p_path, "/CandidateGenerator")
            else:
                p_path = p_path.parent.resolve()
            tries += 1

            if tries == 4:
                log("Error while search the repository for the Data & Models Directories...", 2)
                raise FileNotFoundError
        log(f"set source folder to {p_path}, and data folder to {os.path.join(p_path, 'Data')}", 2)
        return p_path, os.path.join(p_path, "Data"), os.path.join(p_path, "Models")

    def add_path(self, file_type, file_name, extension=None):

        if extension:
            self.files[file_type] = self.repo + '/' + extension + '/' + file_name
        else:
            self.files[file_type] = self.repo + '/' + file_name

    def write_lines(self, file_name, lines):
        file_path = self.repo + file_name

        with open(file_path, "w", encoding="utf-8") as outfile:
            for line in lines:
                outfile.write(line)
        outfile.close()

    def read_aida_input_doc(self, file_key, debug_size=None, file_name=None):
        """
        This method reads the aida training dataset file line by line.
        If matches entry tags (<http://{doc_name}//{id}#{span}>)
          It first will match text entries, and add the text to a list of texts.
          The second entry it matches is of mentions, and then adds the span of the mention to a list.
          Thirdly, it matches the gold entity entry linked to the mention, and appends it to the mention span
          Finally if it matches the entry with a new TXT_ID it sorts the list of spans and tags the mentions in the text
        :param file_key:
        :param debug_size
        :param file_name
        :return:
        """
        mention_count = 0

        try:
            file_name = self.files[file_key][self.files[file_key].rfind("/")+1:]
        except KeyError:
            if file_name is not None and os.path.exists(f"{self.repo}/Datasets/{file_name}"):
                self.add_path(file_key, file_name, extension="Datasets")
            else:
                raise FileNotFoundError

        # _____________ Regular expressions ______________
        # Regex to match new text entry and its id
        entry_tag = re.compile(r'''<http://''' + re.escape(file_name) +
                               r'''/(?P<entry_id>\d*?)#(?P<start_id>\d*?),(?P<end_id>\d*?)>''')
        # Regex to match line that contains the text
        text_regex = re.compile(r''' *nif:isString *"(?P<text>.*)"''')
        # Regex to match line that contains Gold entity
        gold_regex = re.compile(r''' *itsrdf:taIdentRef *<(?P<gold_entity>.*)> .''')

        # ________________ Datastructures to hold results _____________
        texts = {}          # Dictionary holding the mention annotated texts
        eg_dict = {}        # Dictionary containing golden entities of the mentions in text {text_id: [golden_entities]}
        mentions_dict = {}  # Dictionary holding the mentions in the text
        mc = {}             # mention count

        # ________________ Open File _____________________
        log(f"Reading file... {self.files[file_key]}", 1)
        input_doc = open(self.files[file_key], encoding="utf8")
        input_doc_lines = input_doc.readlines()

        # _____________ Control variables
        entry_id = -1
        mention_spans = []

        # ______________ Parse doc
        for line in input_doc_lines:

            entry_match = entry_tag.match(line)
            if entry_match:                                             # If line indicated new text
                entry_id = int(entry_match.group('entry_id'))           # Index of the text in the input file

                if entry_id in texts.keys():                            # If line indicates mention entry
                    # Saving the mention's for later extraction of the embeddings
                    mention_spans.append(
                        (int(entry_match.group('start_id')),
                         int(entry_match.group('end_id')),
                         texts[entry_id][int(entry_match.group('start_id')): int(entry_match.group('end_id'))]
                         )
                    )

                    if mc.get(texts[entry_id][int(entry_match.group('start_id')): int(entry_match.group('end_id'))]):
                        mc[texts[entry_id][int(entry_match.group('start_id')): int(entry_match.group('end_id'))]] += 1
                    else:
                        mc[texts[entry_id][int(entry_match.group('start_id')): int(entry_match.group('end_id'))]] = 1

                elif entry_id > 0:
                    # At the end of parsing the mentions in the text we reset the mentions_spans list and annotate
                    # the text to mark all mentions detected
                    # this is done separately at the end because the mentions entries are not ordered as in the text
                    mention_spans.sort(key=lambda x: x[0])
                    mentions_dict[entry_id - 1] = [msf[2] for msf in mention_spans]
                    eg_dict[entry_id - 1] = [eg[3] for eg in mention_spans]
                    mention_spans = [(span[0], span[1]) for span in mention_spans]
                    texts[entry_id - 1] = annotate_text(texts[entry_id-1], mention_spans)
                    mention_spans = []

                    if debug_size != None:
                        if debug_size == entry_id -1:
                            return texts, eg_dict, mentions_dict

            else:
                text_match = text_regex.match(line)
                if text_match:                                          # Detecting line containing the input text
                    texts[entry_id] = text_match.group('text')
                    continue

                gold_entity_match = gold_regex.match(line)
                if gold_entity_match:                                   # Detecting line containing gold entity link
                    if "notInWiki" in gold_entity_match.group("gold_entity"):
                        mention_spans.pop()
                    else:
                        mention_spans[-1] = (mention_spans[-1][0], mention_spans[-1][1], mention_spans[-1][2],
                                             gold_entity_match.group("gold_entity"))
                        mention_count += 1

        # TODO Add to the comment the format of the return values
        log(f"Extracted {mention_count} mentions from the given Dataset", 1)
        log(f" --> Returning the texts = {{int:tid: str:annotated_text}}, \t eg_dict = {{tid: [str: URL, ]}}, \t"
            f" mention_dict = {{tid: [str: mention_sf, ]}}", 2)

        from operator import itemgetter
        from collections import OrderedDict

        smc = OrderedDict(sorted(mc.items(), key=itemgetter(1)))

        return texts, eg_dict, mentions_dict, smc

    def load_ent_embedding(self):
        ent_emb_file = self.results_repo + f"{self.index_name[:self.index_name.find('_')+1]}entity_embeddings.tsv"

        ent_emb_file = open(ent_emb_file, "r", encoding="utf-8")
        reader = csv.reader(ent_emb_file, delimiter="\t")
        ent_emb_index = {}

        for row in reader:
            if len(row) == 2:
                emb = torch.FloatTensor([float(elem) for elem in row[1][7:].strip("[]()").split(",")])
                ent_emb_index[row[0]] = emb

        return ent_emb_index

    def write_results(self, pairs, corpus, pairs_file_name, corpus_file_name):
        out_file_path = self.repo + "/Results/"

        ent_index = self.load_ent_embedding()

        pairs_path = out_file_path + pairs_file_name
        with open(pairs_path, 'w', encoding='utf8') as out_file:
            writer = csv.writer(out_file, delimiter="\t")
            for pair in pairs:
                tid = pair[0]
                gold_ent = pair[1]
                mention = pair[2]
                try:
                    ent_emb = ent_index[gold_ent]
                except KeyError:
                    try:
                        gold_ent = pair[2]
                        mention = pair[1]
                        ent_emb = ent_index[gold_ent]
                    except KeyError:
                        continue
                writer.writerow([tid, gold_ent, mention, pair[3].numpy(), ent_emb.numpy()])
        out_file.close()

        corpus_path = out_file_path + corpus_file_name
        if not os.path.exists(corpus_path) and corpus:
            with open(corpus_path, "w+") as out_file:
                for key in corpus:
                    for text in range(len(corpus[key])):
                        out_file.write(f"Text#{key}#{text}\n")
                        text = " ".join([w for w in corpus[key][text]])
                        out_file.write(text)
                        out_file.write("\n")
            out_file.close()

    def read_tsv_embeddings_file(self, file_name):

        file = self.repo + "/Results/" + file_name

        log(f"Reading tsv_file {file}", 0)
        data = open(file, 'r')
        reader = csv.reader(data, delimiter='\t')

        es_data_dict = {}

        for row in reader:
            embedding = []
            if len(row) > 0:
                emb = row[1].strip('[').strip(']').split(' ')
                for val in emb:
                    try:
                        embedding.append(float(val))
                    except ValueError:
                        pass
                es_data_dict[row[0]] = embedding

        data.close()
        return es_data_dict

    def read_csv_embeddings_file(self, file_name):
        file_path = self.repo + "/Datasets/Shallom_entity_embeddings.csv"

        start = time.time()
        log(f"Reading tsv_file {file_path} at {time_format(start)}", 0)
        data = open(file_path, 'r')
        reader = csv.reader(data, delimiter=',')

        es_data_dict = {}
        first_line = True

        lines = 0

        for row in reader:
            lines += 1
            if lines == 10000:
                break
            if len(row) > 0 and not first_line:
                embedding = [float(x) for x in row[1:]]
                es_data_dict[row[0]] = embedding
            else:
                first_line = False

        data.close()
        log(f"Finished reading the KGE file in {time_format(time.time() - start)} seconds")
        return es_data_dict

    @staticmethod
    def read_data_pairs_file(file_path):

        log(f"Reading tsv_file {file_path}", 0)
        data = open(file_path, 'r', encoding="utf-8")
        reader = csv.reader(data, delimiter='\t')

        pairs = []

        for row in reader:
            if len(row) > 0:
                txt_id = row[0]
                mention = row[2]
                entity = row[1]
                m_emb = str_to_tensor(row[3], 768)
                e_emb = str_to_tensor(row[4], 25)

                pairs.append((str_to_tuple(txt_id), mention, entity, m_emb, e_emb))

        return pairs

    def load_specific_data(self, file_path, top=0):
        count = {}
        file = open(file_path, "r", encoding="utf-8")
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) > 1:
                eg = row[1]

                if eg in count.keys():
                    count[eg] += 1
                else:
                    count[eg] = 1

        if top != 0:
            count = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
            count = dict(list(count.items())[-top:])

        train_dataset = []
        path = file_path
        train_data_file = open(path, "r", encoding="utf-8")

        csv_reader = csv.reader(train_data_file, delimiter="\t")
        for row in csv_reader:
            if len(row) > 1:
                # if row[1] in count.keys() and count[row[1]] > 2:
                if row[1] in count.keys():
                    txt_id = row[0]
                    mention = row[2]
                    entity = row[1]
                    m_emb = str_to_tensor(row[3], 768)

                    train_dataset.append((str_to_tuple(txt_id), entity, mention, m_emb))

        train_data_file.close()
        return train_dataset

