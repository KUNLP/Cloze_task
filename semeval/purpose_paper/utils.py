from typing import Optional, List

import pandas as pd
from transformers import PreTrainedTokenizer
import json
from purpose_paper.define_class import InputExample, InputFeature


# multi oped 용
import csv,random
import torch
from transformers import BartTokenizer, T5Tokenizer



def convert_examples_to_features(
        examples: List[InputExample],
        tokenizer,
        args
):

    '''
    batch_encoding = tokenizer(
        [(example.prev_sentence+example.now_sentence+example.next_sentence) for example in examples],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeature(**inputs)
        features.append(feature)
    return features
    '''

def load_labelset(
        stage:Optional[str]
):
    label_str_to_int_dic = {'PLAUSIBLE': 0, 'NEUTRAL': 1, 'IMPLAUSIBLE': 2}

    if 'train' in stage:
        labelset = pd.read_csv('../data/train/trainlabels.tsv', sep='\t', header=None, names=['Id', "Label"])
        labelset = list(labelset["Label"])
        labelset = [label_str_to_int_dic[label] for label in labelset]
        return labelset
    elif 'val' in stage:
        labelset = pd.read_csv('../data/dev/devlabels.tsv', sep='\t', header=None, names=['Id', "Label"])
        labelset = list(labelset["Label"])
        labelset = [label_str_to_int_dic[label] for label in labelset]
        return labelset


# multioped용
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        #print(self.encodings.items()[0])
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #print(self.labels[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_csv(path, tok_type="bart"):
    """loading the csv data file"""

    source = []
    target = []
    query = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # print(row)
            query.append(row[0])
            source.append(row[4])
            target.append(row[3])

    source = source[1:]
    target = target[1:]
    query = query[1:]

    # print(len(source))
    # print(len(target))
    # print(len(query))

    for i in range(len(source)):
        source[i] = source[i].replace('\n', '')
        target[i] = target[i].replace('\n', '')
        query[i] = query[i].replace('\n', '')

    # print(len(total_texts))
    # randomize the train/dev/test/ split
    total_texts = [(source[i], source[i + 1], query[i]) for i in range(0, len(source) - 1, 2)]
    total_labels = [(target[i], target[i + 1], query[i]) for i in range(0, len(target) - 1, 2)]
    # print(total_texts[:3])
    # print(total_labels[:3])

    # print(total_query[:3])
    random.Random(4).shuffle(total_texts)
    random.Random(4).shuffle(total_labels)
    # random.Random(4).shuffle(total_query)
    # print(total_texts[:3])
    # print(total_labels[:3])
    # print(len(total_texts))
    # print(total_query[:3])

    train_len = len(total_texts) * 7 // 10
    dev_len = len(total_texts) * 8 // 10
    # print(train_len)

    train_texts = []
    train_labels = []
    train_query = []

    dev_texts = []
    dev_labels = []
    dev_query = []

    test_texts = []
    test_labels = []
    test_query = []

    for i in range(train_len):
        train_texts.append(total_texts[i][0])
        train_texts.append(total_texts[i][1])
        train_labels.append(total_labels[i][0])
        train_labels.append(total_labels[i][1])
        train_query.append(total_texts[i][2])
        train_query.append(total_labels[i][2])

    for i in range(train_len, dev_len):
        dev_texts.append(total_texts[i][0])
        dev_texts.append(total_texts[i][1])
        dev_labels.append(total_labels[i][0])
        dev_labels.append(total_labels[i][1])
        dev_query.append(total_texts[i][2])
        dev_query.append(total_labels[i][2])

    for i in range(dev_len, len(total_texts)):
        test_texts.append(total_texts[i][0])
        test_texts.append(total_texts[i][1])
        test_labels.append(total_labels[i][0])
        test_labels.append(total_labels[i][1])
        test_query.append(total_texts[i][2])
        test_query.append(total_labels[i][2])

    dic = {}
    for i in range(len(train_labels)):
        dic[train_labels[i]] = train_query[i]
        # if train_query[i]== "was trump right to kill soleimani?":
        #    print("here", train_labels[i])

    for i in range(len(dev_labels)):
        dic[dev_labels[i]] = dev_query[i]

    for i in range(len(test_labels)):
        dic[test_labels[i]] = test_query[i]

    if tok_type == "bart":
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    else:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

    train_encodings = tokenizer(train_query, text_pair=train_texts)
    train_label_encodings = tokenizer(train_labels)['input_ids']
    train_dataset = Dataset(train_encodings, train_label_encodings)

    dev_encodings = tokenizer(dev_query, text_pair=dev_texts)
    dev_label_encodings = tokenizer(dev_labels)['input_ids']
    dev_dataset = Dataset(dev_encodings, dev_label_encodings)

    test_encodings = tokenizer(test_query, text_pair=test_texts)
    test_label_encodings = tokenizer(test_labels)['input_ids']
    test_dataset = Dataset(test_encodings, test_label_encodings)

    # print(train_dataset[0])
    # print(train_dataset[0]['input_ids'])
    # print(tokenizer.decode(train_dataset[0]['input_ids']))
    # print(tokenizer.decode(train_dataset[0]['labels']))

    return train_dataset, dev_dataset, test_dataset, dic


if __name__ == "__main__":
    train_dataset, dev_dataset, test_dataset, dic = load_csv('../data/MultiOpEd.csv')