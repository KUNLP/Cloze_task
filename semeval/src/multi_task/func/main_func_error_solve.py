import json, argparse
import spacy
import os
from itertools import product

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import ElectraTokenizerFast, ElectraConfig

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd

from src.multi_task.model.model import ElectraForMultiTaskClassification
from src.multi_task.func.my_utils import retrieve_all_instances_from_dataset, retrieve_labels_from_dataset_for_classification
from src.multi_task.func.grounding_concept import read_concept_vocab, load_matcher
from src.multi_task.func.multitask_utils_final import read_my_multi_dataset, convert_multidata2tensordataset, convert_multitask_dataset_dev_to_tensordataset
from src.multi_task.func.make_multitask_dataset_final import retrieve_all_processed_data_from_dataset


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def find_tok_start_end_position(tokenizer, doc_tokens, start_position, end_position, answer_span=None):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    tok_start_position = orig_to_tok_index[start_position]
    if end_position < len(doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[end_position + 1] - 1
    else:
        tok_end_position = len(all_doc_tokens) - 1
    for new_start in range(tok_start_position, tok_end_position + 1):
        for new_end in range(tok_end_position, new_start -1, -1):
            text_span = " ".join(doc_tokens[new_start:new_end+1])
            if text_span == answer_span:
                return new_start, new_end
    return tok_start_position, tok_end_position


def downsampling(pairs_idx_list, relation_exist_list, relation_type_list):
    ratio = 4


    if num_exist is 0:
        return pairs_idx_list, relation_exist_list, relation_type_list
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--path_to_train', type=str, default='../../../data/train/traindata.tsv')
    parser.add_argument('--path_to_train_labels', type=str, default='../../../data/train/trainlabels.tsv')
    parser.add_argument('--path_to_dev', type=str, default='../../../data/dev/devdata.tsv')
    parser.add_argument('--path_to_dev_labels', type=str, default='../../../data/dev/devlabels.tsv')


    parser.add_argument('--path_to_multitask_dataset_train', type=str, default='../../../data/multitask_dataset_train.json')
    parser.add_argument('--path_to_multitask_dataset_dev', type=str, default='../../../data/multitask_dataset_dev.json')
    parser.add_argument('--path_to_conceptnet', type=str, default='../../../conceptnet/assertions-570-en.csv')
    parser.add_argument('--path_to_lemma_json', type=str, default='../../../conceptnet/lemma_matching.json')
    parser.add_argument('--path_to_concept_word', type=str, default='../../../conceptnet/concept_word.txt')

    parser.add_argument('--plausible_num_label', type=int, default=3)
    parser.add_argument('--relation_exist_label', type=int, default=2)
    parser.add_argument('--relation_type_label', type=int, default=18)

    parser.add_argument('--input_sentence_max_length', type=int, default=256) # title 과 문장 다 합친거 토크나이즈 했을때 최대 길이 256
    parser.add_argument('--max_length_pair_token', type=int, default=159) #페어쌍으로 했을때 최대길이 159
    parser.add_argument('--relation_exist_padding_idx', type=int, default=2)
    parser.add_argument('--relation_type_padding_idx', type=int, default=18)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    parser.add_argument("--relation2idx", type=dict, default={'no relation': 0, 'antonym': 1, 'atlocation': 2, 'capableof': 3, 'causes': 4, 'createdby': 5, 'desires': 6, 'hascontext': 7, 'hasproperty': 8, 'hassubevent': 9, 'isa': 10, 'madeof': 11, 'notcapableof': 12, 'notdesires': 13, 'partof': 14, 'receivesaction': 15, 'relatedto': 16, 'usedfor': 17})
    parser.add_argument('--idx2relation', type=dict, default={0: 'no relation', 1: 'antonym', 2: 'atlocation', 3: 'capableof', 4: 'causes', 5: 'createdby', 6: 'desires', 7: 'hascontext', 8: 'hasproperty', 9: 'hassubevent', 10: 'isa', 11: 'madeof', 12: 'notcapableof', 13: 'notdesires', 14: 'partof', 15: 'receivesaction', 16: 'relatedto', 17: 'usedfor'})

    parser.add_argument("--output_dir", type=str, default="../../../output")
    parser.add_argument("--PLM", type=str, default='electra-large')
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    electra_tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')

    # 원본 train 데이터셋 로딩
    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    ids, train_titles, train_processed_sentences, train_answers, train_original_answer_char_positions, \
        train_prev_sentences, train_now_sentences, train_next_sentences = retrieve_all_processed_data_from_dataset(train_set)
    train_label_set = pd.read_csv(args.path_to_train_labels, sep='\t', header=None, names=["Id", "Label"])
    train_plausible_labels = retrieve_labels_from_dataset_for_classification(train_label_set)

    dev_set = pd.read_csv(args.path_to_dev, sep='\t', quoting=3)
    dev_ids, dev_titles, dev_processed_sentences, dev_answers, dev_original_answer_char_positions, \
        dev_prev_sentences, dev_now_sentences, dev_next_sentences = retrieve_all_processed_data_from_dataset(dev_set)


    conceptnet_dataset = pd.read_csv(args.path_to_conceptnet, sep='\t', header=None,
                                     names=['relation', 'header', 'tail', 'weight'])
    # 내가 만든 멀티태스크 데이터셋 로딩
    with open(args.path_to_multitask_dataset_dev, 'r', encoding='utf8') as f:
        multi_task_dataset_dev = json.load(f)
    with open(args.path_to_multitask_dataset_train, 'r', encoding='utf8') as f:
        multi_task_dataset_train = json.load(f)

    train_sentence_span_concepts = []
    train_answer_span_concepts = []
    for data in multi_task_dataset_train:
        train_sentence_span_concepts.append(data['sentence_span_concept'])
        train_answer_span_concepts.append(data['answer_span_concept'])

    for idx, (train_answer_span_concept, answer) in enumerate(zip(train_answer_span_concepts, train_answers)):
        if not train_answer_span_concept:
            if len(answer.split(" ")) > 1:
                train_answer_span_concepts[idx] = [answer.split(" ")[1].lower()]
            else:
                train_answer_span_concepts[idx] = [answer.lower()]
        else:
            train_answer_span_concepts[idx] = [answer_span.lower() for answer_span in train_answer_span_concept]

    dev_sentence_span_concepts = []
    dev_answer_span_concepts = []
    for data in multi_task_dataset_dev:
        dev_sentence_span_concepts.append(data['sentence_span_concept'])
        dev_answer_span_concepts.append(data['answer_span_concept'])
    for idx, (dev_answer_span_concept, answer) in enumerate(zip(dev_answer_span_concepts, dev_answers)):
        if not dev_answer_span_concept:
            if len(answer.split(" ")) > 1:
                dev_answer_span_concepts[idx] = [answer.split(" ")[1].lower()]
            else:
                dev_answer_span_concepts[idx] = [answer.lower()]
        else:
            dev_answer_span_concepts[idx] = [answer_span.lower() for answer_span in dev_answer_span_concept]


    #컨셉 product 랭스 맥스 값 구하기
    max_pair_length = 0
    for sentence_span, answer_span in zip(train_sentence_span_concepts, train_answer_span_concepts):
        pairs = [sentence_span, answer_span]
        pairs = list(product(*pairs))
        if max_pair_length < len(pairs):
            max_pair_length = len(pairs)
            print(pairs)
    print("train_pair max length : {}".format(max_pair_length))

    max_pair_length = 0
    for sentence_span, answer_span in zip(dev_sentence_span_concepts, dev_answer_span_concepts):
        pairs = [sentence_span, answer_span]
        pairs = list(product(*pairs))
        if max_pair_length < len(pairs):
            max_pair_length = len(pairs)
            print(pairs)
    print("dev_pair max length : {}".format(max_pair_length))


    for idx, (train_title, train_processed_sentence, train_answer, train_original_answer_char_position, train_sentence_span_concept, train_answer_span_concept) in \
            tqdm(enumerate(zip(train_titles, train_processed_sentences, train_answers, train_original_answer_char_positions, train_sentence_span_concepts, train_answer_span_concepts)), desc='convert dataset', total=len(train_titles)):

        processed_title = electra_tokenizer(train_title)
        processed_sentence = electra_tokenizer(train_processed_sentence)

        input_ids = processed_title['input_ids'] + processed_sentence['input_ids'][1:]
        attention_mask = processed_title['attention_mask'] + processed_sentence['attention_mask'][1:]

        assert len(input_ids) == len(attention_mask)
        padding_input_ids = [0] * (args.input_sentence_max_length - len(input_ids))
        input_ids += padding_input_ids
        attention_mask += padding_input_ids
        assert len(input_ids) == len(attention_mask)

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in train_processed_sentence:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        pairs = [train_sentence_span_concept, train_answer_span_concept]
        pairs = list(product(*pairs))
        pairs_idx_list = []
        relation_exist = []
        relation_type = []
        temp_conceptnet = conceptnet_dataset[(conceptnet_dataset['header'].isin(train_sentence_span_concept))|(conceptnet_dataset['tail'].isin(train_answer_span_concept))]
        for pair in pairs:
            sentence_concept, answer_concept = pair[0], pair[1]
            sen_start_position = char_to_word_offset[train_processed_sentence.find(sentence_concept)]
            sen_end_position = char_to_word_offset[min(train_processed_sentence.find(sentence_concept)+len(sentence_concept)-1, len(char_to_word_offset)-1)]
            ans_start_position = char_to_word_offset[train_processed_sentence.find(answer_concept)]
            ans_end_position = char_to_word_offset[min(train_processed_sentence.find(answer_concept)+len(answer_concept)-1, len(char_to_word_offset)-1)]

            sen_tok_start, sen_tok_end = find_tok_start_end_position(electra_tokenizer, doc_tokens, sen_start_position, sen_end_position)
            ans_tok_start, ans_tok_end = find_tok_start_end_position(electra_tokenizer, doc_tokens, ans_start_position, ans_end_position, answer_concept)

            if sen_tok_start > sen_tok_end or ans_tok_start > ans_tok_end:
                print("processed_sentence : {}".format(train_processed_sentence))
                print("pair : {}".format(pair))
                print("sen tok start : {}, tok end : {}".format(sen_tok_start, sen_tok_end))
                print("ans tok start : {}, tok end : {}".format(ans_tok_start, ans_tok_end))

            pairs_idx_list.append([
                [len(processed_title['input_ids']) + sen_tok_start, len(processed_title['input_ids']) + sen_tok_end + 1],
                [len(processed_title['input_ids']) + ans_tok_start, len(processed_title['input_ids']) + ans_tok_end + 1]
            ])
            condition = (temp_conceptnet['header'] == sentence_concept.replace(' ', '_')) & (
                        temp_conceptnet['tail'] == answer_concept.replace(' ', '_'))
            if not temp_conceptnet[condition].empty:
                relation_exist.append(1)
                relation_type.append(
                    args.relation2idx[temp_conceptnet[condition]['relation'].values[0]])  # dict2idx 이런식으로 바꿔넣자
            else:
                relation_exist.append(0)
                relation_type.append(0)

        pair_padding = [[-1, -1], [-1, -1]]
        for i in range(args.max_length_pair_token - len(pairs_idx_list)):
            pairs_idx_list.append(pair_padding)
            relation_exist.append(args.relation_exist_padding_idx)
            relation_type.append(args.relation_type_padding_idx)

        if idx < 10:
            print()
            print(" *** EXAMPLE ***")
            print("title : {}".format(train_title))
            print("sentence : {}".format(train_processed_sentence))
            print("input_ids : {}".format(input_ids))
            print("attention_mask : {}".format(attention_mask))
            print("sentence concept : {}".format(train_sentence_span_concept))
            print("answer concept : {}".format(train_answer_span_concept))
            print("pair index list length : {}, list : {}".format(len(pairs_idx_list), pairs_idx_list))
            print("pair sentence concept decode : {}".format(
                [electra_tokenizer.decode(input_ids[pair_idx[0][0]:pair_idx[0][1]]) for pair_idx in pairs_idx_list]
            ))
            print("pair answer concept decode : {}".format(
                [electra_tokenizer.decode(input_ids[pair_idx[1][0]:pair_idx[1][1]]) for pair_idx in pairs_idx_list],
            ))
            print("relation exist length : {}, list : {}".format(len(relation_exist), relation_exist))
            print("relation type length : {}, list : {}".format(len(relation_type), relation_type))
            print()