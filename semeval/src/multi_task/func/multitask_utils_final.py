import json, argparse, re, logging
import pandas as pd
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import TensorDataset
from transformers import ElectraTokenizer

import spacy
from itertools import product

from src.multi_task.func.grounding_concept import load_matcher
from src.multi_task.func.my_utils import retrieve_all_instances_from_dataset
from src.multi_task.func.my_utils import retrieve_labels_from_dataset_for_classification
from src.multi_task.func.grounding_concept import read_concept_vocab
logging.basicConfig(level=logging.DEBUG)


# 논문에선 어차피 dev 데이터셋으로 하니까 나중에 dev로 test 할 필요가 있긴함
def load_pickle_data(args, func_mode):
    if args.mode is 'train' and func_mode is 'train':
        with open('../../data/save_pickle_train_data.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.mode is 'train' and func_mode is 'dev':
        with open('../../data/save_pickle_dev_data.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        with open('../../data/save_pickle_test_data.pkl', 'rb') as f:
            data = pickle.load(f)
    total_input_ids = torch.tensor(data['total_input_ids'], dtype=torch.long)
    total_attention_mask = torch.tensor(data['total_attention_mask'], dtype=torch.long)
    if args.mode is not 'test':
        total_plausible_label = torch.tensor(data['total_plausible_label'], dtype=torch.long)
    total_pair_ids = torch.tensor(data['total_pair_ids'], dtype=torch.long)
    total_relation_exist_label = torch.tensor(data['total_relation_exist_label'], dtype=torch.long)
    total_relation_type_label = torch.tensor(data['total_relation_type_label'], dtype=torch.long)
    if args.mode is not 'test':
        dataset = TensorDataset(total_input_ids, total_attention_mask, total_plausible_label,
                                total_pair_ids, total_relation_exist_label, total_relation_type_label)
    else:
        dataset = TensorDataset(total_input_ids, total_attention_mask, total_pair_ids, total_relation_exist_label, total_relation_type_label)
    return dataset



def standard_sentence(sentence):
    sentence = sentence.replace("don't", "do not")
    sentence = sentence.replace("—", " ")
    sentence = sentence.replace('\"', '')
    sentence = sentence.replace("(...)", "")
    sentence = sentence.replace("*", "")
    sentence = sentence.replace(";", "")
    sentence = sentence.lower()
    return sentence.strip()


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

# def find_tok_start_end_position(tokenizer, doc_tokens, start_position, end_position):
#     tok_to_orig_index = []
#     orig_to_tok_index = []
#     all_doc_tokens = []
#     for (i, token) in enumerate(doc_tokens):
#         orig_to_tok_index.append(len(all_doc_tokens))
#         sub_tokens = tokenizer.tokenize(token)
#         for sub_token in sub_tokens:
#             tok_to_orig_index.append(i)
#             all_doc_tokens.append(sub_token)
#     tok_start_position = orig_to_tok_index[start_position]
#     if end_position < len(doc_tokens) - 1:
#         tok_end_position = orig_to_tok_index[end_position + 1] - 1
#     else:
#         tok_end_position = len(all_doc_tokens) - 1
#
#     return tok_start_position, tok_end_position



def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_", " "))
    lemm = set()
    lemm.add("_".join([token.lemma_ for token in doc]))
    return lemm


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def read_my_multi_dataset(dataset, relation2idx):
    sentences = []
    answers = []
    relation_exists = []
    relation_types = []
    sentences_concepts = []
    answers_concepts = []
    concept_pairs = []

    for data in dataset:
        sentence = data['context']
        sentence_concepts = data['context_concepts']

        answer = data['answers']
        answer_concepts = data['answers_concepts']

        concept_pair = data['concept_pairs']
        relation_exist = data['relation_exist']
        relation_type = data['relation_type']

        # 정답개수에 맞게 데이터 쪼갬
        for option in answer:
            context = sentence.replace("______", option)
            sentences.append(context)
            answers.append(option)
            relation_exists.append(relation_exist[option])

            relation2num = []
            for relation in relation_type[option]:
                if relation !=0:
                    relation2num.append(relation2idx[relation])
                else:
                    relation2num.append(0)
            relation_types.append(relation2num)

            sentences_concepts.append(sentence_concepts)
            answers_concepts.append(answer_concepts[option])
            concept_pairs.append(concept_pair[option])

    return sentences, sentences_concepts, answers, answers_concepts, relation_exists, relation_types, concept_pairs


#테스트용
def convert_all_instance_to_tensordataset_test(
        args, tokenizer, conceptnet_dataset,
        titles_instances, processed_sentence_instances, answer_instances, sentence_concepts, answer_concepts
):
    total_input_ids = []
    total_attention_mask = []
    total_pair_ids = []
    total_relation_exist_label = []
    total_relation_type_label = []

    for idx, (title, sentence, answer, sentence_concept, answer_concept) in \
            tqdm(enumerate(zip(titles_instances, processed_sentence_instances, answer_instances, sentence_concepts, answer_concepts)),
                 total=len(titles_instances), desc='convert tensordataset'):

        processed_title = tokenizer(title)
        processed_sentence = tokenizer(sentence)
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
        for c in sentence:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        pairs = [sentence_concept, answer_concept]
        pairs = list(product(*pairs))
        pairs_idx_list = []
        relation_exist_label = []
        relation_type_label = []
        temp_conceptnet = conceptnet_dataset[(conceptnet_dataset['header'].isin(sentence_concept)) | (
            conceptnet_dataset['tail'].isin(answer_concept))]

        for pair in pairs:
            sen_concept, ans_concept = pair[0], pair[1]
            sen_start_position = char_to_word_offset[sentence.find(sen_concept)]
            sen_end_position = char_to_word_offset[min(sentence.find(sen_concept) + len(sen_concept) -1, len(char_to_word_offset) -1 )]
            ans_start_position = char_to_word_offset[sentence.find(ans_concept)]
            ans_end_position = char_to_word_offset[min(sentence.find(ans_concept) + len(ans_concept) -1, len(char_to_word_offset) -1)]

            sen_tok_start, sen_tok_end = find_tok_start_end_position(tokenizer, doc_tokens, sen_start_position,
                                                                     sen_end_position)
            ans_tok_start, ans_tok_end = find_tok_start_end_position(tokenizer, doc_tokens, ans_start_position,
                                                                     ans_end_position, answer_concept)
            if sen_tok_start > sen_tok_end or ans_tok_start > ans_tok_end:
                print("!!! error !!!")
                print("processed sentence : {}".format(processed_sentence))
                print("problem pair : {}".format(pair))
                print("sen tok start : {}, tok end : {}".format(sen_tok_start, sen_tok_end))
                print("ans tok start : {}, tok end : {}".format(ans_tok_start, ans_tok_end))

            pairs_idx_list.append([
                [len(processed_title['input_ids']) + sen_tok_start, len(processed_title['input_ids']) + sen_tok_end + 1],
                [len(processed_title['input_ids']) + ans_tok_start, len(processed_title['input_ids']) + ans_tok_end + 1]
            ])
            condition = (temp_conceptnet['header'] == sen_concept.replace(' ', '_')) & (
                    temp_conceptnet['tail'] == ans_concept.replace(' ', '_'))
            if not temp_conceptnet[condition].empty:
                relation_exist_label.append(1)
                relation_type_label.append(
                    args.relation2idx[temp_conceptnet[condition]['relation'].values[0]])  # dict2idx 이런식으로 바꿔넣자
            else:
                relation_exist_label.append(0)
                relation_type_label.append(0)

        pair_padding = [[-1, -1], [-1, -1]]
        for i in range(args.max_length_pair_token - len(pairs_idx_list)):
            pairs_idx_list.append(pair_padding)
            relation_exist_label.append(args.relation_exist_padding_idx)
            relation_type_label.append(args.relation_type_padding_idx)

        if idx < 2:
            print()
            print(" *** EXAMPLE ***")
            print("title : {}".format(title))
            print("sentence : {}".format(processed_sentence))
            print("input_ids : {}".format(input_ids))
            print("attention_mask : {}".format(attention_mask))
            print("sentence concept : {}".format(sentence_concept))
            print("answer concept : {}".format(answer_concept))
            print("pair index list length : {}, list : {}".format(len(pairs_idx_list), pairs_idx_list))
            print("pair sentence concept decode : {}".format(
                [tokenizer.decode(input_ids[pair_idx[0][0]:pair_idx[0][1]]) for pair_idx in pairs_idx_list]
            ))
            print("pair answer concept decode : {}".format(
                [tokenizer.decode(input_ids[pair_idx[1][0]:pair_idx[1][1]]) for pair_idx in pairs_idx_list],
            ))
            print("relation exist length : {}, list : {}".format(len(relation_exist_label), relation_exist_label))
            print("relation type length : {}, list : {}".format(len(relation_type_label), relation_type_label))
            print()

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_pair_ids.append(pairs_idx_list)
        total_relation_exist_label.append(relation_exist_label)
        total_relation_type_label.append(relation_type_label)

        # if idx == 99:
        #     break
    # 나중엔 그냥 피클 로드해서 바롤 돌리면 될듯
    data_dict = {}
    data_dict['total_input_ids'] = total_input_ids
    data_dict['total_attention_mask'] = total_attention_mask
    data_dict['total_pair_ids'] = total_pair_ids
    data_dict['total_relation_exist_label'] = total_relation_exist_label
    data_dict['total_relation_type_label'] = total_relation_type_label

    with open(args.save_pickle_test_data, 'wb') as f:
        pickle.dump(data_dict, f)
    #
    # if make_train:
    #     with open(args.save_pickle_train_data, 'wb') as f:
    #         pickle.dump(data_dict, f)
    # else:
    #     with open(args.save_pickle_dev_data, 'wb') as f:
    #         pickle.dump(data_dict, f)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_pair_ids = torch.tensor(total_pair_ids, dtype=torch.long)
    total_relation_exist_label = torch.tensor(total_relation_exist_label, dtype=torch.float)
    total_relation_type_label = torch.tensor(total_relation_type_label, dtype=torch.long)
    dataset = TensorDataset(total_input_ids, total_attention_mask,
                            total_pair_ids, total_relation_exist_label, total_relation_type_label)
    return dataset

# 1/21 title까지 합친거
def convert_all_instance_to_tensordataset(
        args, tokenizer, conceptnet_dataset,
        titles_instances, processed_sentence_instances, answer_instances, sentence_concepts, answer_concepts, plausible_lables,
        make_train
):
    total_input_ids = []
    total_attention_mask = []
    total_plausible_label = []
    total_pair_ids = []
    total_relation_exist_label = []
    total_relation_type_label = []

    for idx, (title, sentence, answer, sentence_concept, answer_concept, plausible_label) in \
            tqdm(enumerate(zip(titles_instances, processed_sentence_instances, answer_instances, sentence_concepts, answer_concepts, plausible_lables)),
                 total=len(titles_instances), desc='convert tensordataset'):

        processed_title = tokenizer(title)
        processed_sentence = tokenizer(sentence)
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
        for c in sentence:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        pairs = [sentence_concept, answer_concept]
        pairs = list(product(*pairs))
        pairs_idx_list = []
        relation_exist_label = []
        relation_type_label = []
        temp_conceptnet = conceptnet_dataset[(conceptnet_dataset['header'].isin(sentence_concept)) | (
            conceptnet_dataset['tail'].isin(answer_concept))]

        for pair in pairs:
            sen_concept, ans_concept = pair[0], pair[1]
            sen_start_position = char_to_word_offset[sentence.find(sen_concept)]
            sen_end_position = char_to_word_offset[min(sentence.find(sen_concept) + len(sen_concept) -1, len(char_to_word_offset) -1 )]
            ans_start_position = char_to_word_offset[sentence.find(ans_concept)]
            ans_end_position = char_to_word_offset[min(sentence.find(ans_concept) + len(ans_concept) -1, len(char_to_word_offset) -1)]

            sen_tok_start, sen_tok_end = find_tok_start_end_position(tokenizer, doc_tokens, sen_start_position,
                                                                     sen_end_position)
            ans_tok_start, ans_tok_end = find_tok_start_end_position(tokenizer, doc_tokens, ans_start_position,
                                                                     ans_end_position, answer_concept)
            if sen_tok_start > sen_tok_end or ans_tok_start > ans_tok_end:
                print("!!! error !!!")
                print("processed sentence : {}".format(processed_sentence))
                print("problem pair : {}".format(pair))
                print("sen tok start : {}, tok end : {}".format(sen_tok_start, sen_tok_end))
                print("ans tok start : {}, tok end : {}".format(ans_tok_start, ans_tok_end))

            pairs_idx_list.append([
                [len(processed_title['input_ids']) + sen_tok_start, len(processed_title['input_ids']) + sen_tok_end + 1],
                [len(processed_title['input_ids']) + ans_tok_start, len(processed_title['input_ids']) + ans_tok_end + 1]
            ])
            condition = (temp_conceptnet['header'] == sen_concept.replace(' ', '_')) & (
                    temp_conceptnet['tail'] == ans_concept.replace(' ', '_'))
            if not temp_conceptnet[condition].empty:
                relation_exist_label.append(1)
                relation_type_label.append(
                    args.relation2idx[temp_conceptnet[condition]['relation'].values[0]])  # dict2idx 이런식으로 바꿔넣자
            else:
                relation_exist_label.append(0)
                relation_type_label.append(0)

        pair_padding = [[-1, -1], [-1, -1]]
        for i in range(args.max_length_pair_token - len(pairs_idx_list)):
            pairs_idx_list.append(pair_padding)
            relation_exist_label.append(args.relation_exist_padding_idx)
            relation_type_label.append(args.relation_type_padding_idx)

        if idx < 2:
            print()
            print(" *** EXAMPLE ***")
            print("title : {}".format(title))
            print("sentence : {}".format(processed_sentence))
            print("input_ids : {}".format(input_ids))
            print("attention_mask : {}".format(attention_mask))
            print("sentence concept : {}".format(sentence_concept))
            print("answer concept : {}".format(answer_concept))
            print("pair index list length : {}, list : {}".format(len(pairs_idx_list), pairs_idx_list))
            print("pair sentence concept decode : {}".format(
                [tokenizer.decode(input_ids[pair_idx[0][0]:pair_idx[0][1]]) for pair_idx in pairs_idx_list]
            ))
            print("pair answer concept decode : {}".format(
                [tokenizer.decode(input_ids[pair_idx[1][0]:pair_idx[1][1]]) for pair_idx in pairs_idx_list],
            ))
            print("relation exist length : {}, list : {}".format(len(relation_exist_label), relation_exist_label))
            print("relation type length : {}, list : {}".format(len(relation_type_label), relation_type_label))
            print()

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_plausible_label.append(plausible_label)
        total_pair_ids.append(pairs_idx_list)
        total_relation_exist_label.append(relation_exist_label)
        total_relation_type_label.append(relation_type_label)

        # if idx == 99:
        #     break
    # 나중엔 그냥 피클 로드해서 바롤 돌리면 될듯
    # data_dict = {}
    # data_dict['total_input_ids'] = total_input_ids
    # data_dict['total_attention_mask'] = total_attention_mask
    # data_dict['total_plausible_label'] = total_plausible_label
    # data_dict['total_pair_ids'] = total_pair_ids
    # data_dict['total_relation_exist_label'] = total_relation_exist_label
    # data_dict['total_relation_type_label'] = total_relation_type_label
    #
    # if make_train:
    #     with open(args.save_pickle_train_data, 'wb') as f:
    #         pickle.dump(data_dict, f)
    # else:
    #     with open(args.save_pickle_dev_data, 'wb') as f:
    #         pickle.dump(data_dict, f)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_plausible_label = torch.tensor(total_plausible_label, dtype=torch.long)
    total_pair_ids = torch.tensor(total_pair_ids, dtype=torch.long)
    total_relation_exist_label = torch.tensor(total_relation_exist_label, dtype=torch.float)
    total_relation_type_label = torch.tensor(total_relation_type_label, dtype=torch.long)
    dataset = TensorDataset(total_input_ids, total_attention_mask, total_plausible_label,
                            total_pair_ids, total_relation_exist_label, total_relation_type_label)
    return dataset


# 지금 이걸 사용하고있다. 21/1/13
def convert_multitask_dataset_dev_to_tensordataset(
        args, tokenizer, conceptnet_dataset,
        prev_sentences_dev, now_sentences_dev, next_sentences_dev, answers_dev,
        sentence_span_concepts, answer_span_concepts, plausible_labels
):
    total_input_ids = []
    total_attention_mask = []
    total_concept_pair_ids = []
    total_relation_exist = []
    total_relation_type = []
    total_plausible_label = []
    for idx, (prev_sentence, now_sentence, next_sentence, sentence_span_concept, answer_span_concept) in tqdm(
        enumerate(zip(prev_sentences_dev, now_sentences_dev, next_sentences_dev, sentence_span_concepts, answer_span_concepts)),
        total=len(now_sentences_dev), desc='convert_tensordataset'
    ):
        sentence = standard_sentence(prev_sentence + now_sentence + next_sentence)
        processed_sentence = tokenizer(sentence)

        # input_ids, attention_mask
        input_ids = processed_sentence['input_ids']
        attention_mask = processed_sentence['attention_mask']
        tokens = processed_sentence.encodings[0].tokens # list 형태임

        assert len(input_ids) == len(attention_mask)
        padding_input_ids = [0] * (args.input_sentence_max_length - len(input_ids))
        input_ids += padding_input_ids
        attention_mask += padding_input_ids
        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)

        #concept_pair의 ids, relation_exist, relation_type
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in sentence:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        if len(answer_span_concept) > 0:
            pairs = [sentence_span_concept, [answer_span.lower() for answer_span in answer_span_concept]]
        else:
            answer_span_concept = [standard_sentence(answers_dev[idx])]

            pairs = [sentence_span_concept, answer_span_concept]
        pairs = list(product(*pairs))
        pair_list = []
        relation_exist = []
        relation_type = []
        temp_conceptnet = conceptnet_dataset[(conceptnet_dataset['header'].isin(sentence_span_concept))|(conceptnet_dataset['tail'].isin(answer_span_concept))]

        for pair in pairs:
            sentence_concept, answer_concept = pair[0], pair[1]

            sen_start_position = char_to_word_offset[sentence.find(sentence_concept)]
            sen_end_position = char_to_word_offset[min(sentence.find(sentence_concept) + len(sentence_concept) -1, len(char_to_word_offset) -1)]
            ans_start_position = char_to_word_offset[sentence.find(answer_concept)]
            ans_end_position = char_to_word_offset[min(sentence.find(answer_concept) + len(answer_concept) -1, len(char_to_word_offset) -1)]

            sen_tok_start, sen_tok_end = find_tok_start_end_position(tokenizer, doc_tokens, sen_start_position, sen_end_position)
            ans_tok_start, ans_tok_end = find_tok_start_end_position(tokenizer, doc_tokens, ans_start_position, ans_end_position)
            if (ans_tok_start>ans_tok_end):
                print()
                print("!!! error !!!")
                print("{}, {}".format(idx, sentence))
                print("len char_to_word_offset, char_to_word_offset : {}".format(len(char_to_word_offset), char_to_word_offset))
                print("min find : {}, len char to word offset : {}".format(sentence.find(answer_concept) + len(answer_concept) -1, len(char_to_word_offset) -1))
                print("sentence_concept : {}, answer_concept : {}".format(sentence_concept, answer_concept))
                print("sen start position, end position : {}, {}".format(sen_start_position, sen_end_position))
                print("sen tok start : {}, tok end : {}".format(sen_tok_start, sen_tok_end))
                print("ans start position, end position : {}, {}".format(ans_start_position, ans_end_position))
                print("ans tok start : {}, tok end : {}".format(ans_tok_start, ans_tok_end))

            pair_list.append([[sen_tok_start+1, sen_tok_end+1], [ans_tok_start+1, ans_tok_end+1]]) # +1은 CLS 때문에 해주는것 나중에 title 추가하면 +2 로 해야할듯

            condition = (temp_conceptnet['header'] == sentence_concept.replace(' ', '_')) & (temp_conceptnet['tail'] == answer_concept.replace(' ', '_'))
            if not temp_conceptnet[condition].empty:
                relation_exist.append(1)
                relation_type.append(args.relation2idx[temp_conceptnet[condition]['relation'].values[0]]) # dict2idx 이런식으로 바꿔넣자
            else:
                relation_exist.append(0)
                relation_type.append(0)



        pair_padding = [[-1, -1], [-1, -1]]
        # args.exist padding idx = 2
        # args.type padding idx = 18
        for i in range(args.max_length_pair_token - len(pair_list)):
            pair_list.append(pair_padding)
            relation_exist.append(args.relation_exist_padding_idx)
            relation_type.append(args.relation_type_padding_idx)

        total_concept_pair_ids.append(pair_list)
        total_relation_exist.append(relation_exist)
        total_relation_type.append(relation_type)
        total_plausible_label.append(plausible_labels[idx])

        if idx < 5:
            print()
            print(" *** Example *** ")
            print("sentence tokens length : {}".format(len(tokens)))
            print("sentence : {}".format(sentence))
            print("sentence concept : {}".format(sentence_span_concept))
            print("answer : {}".format(answers_dev[idx]))
            print("answer concept : {}".format(answer_span_concept))
            print("input_ids : {}".format(input_ids))
            print("attention_mask : {}".format(attention_mask))
            print("concept_pair : {}".format([pair for pair in pair_list if pair != [[-1, -1], [-1, -1]]]))
            print("concept_pair_token : {}, {}".format(tokens[pair_list[idx][0][0]:pair_list[idx][0][1]+1], tokens[pair_list[idx][1][0]:pair_list[idx][1][1]+1]))
            print("pairs : {}, {}".format(pairs[idx][0], pairs[idx][1]))
            print("relation_exist : {}".format([exist for exist in relation_exist if exist != 2]))
            print("relation_type : {}".format([rtype for rtype in relation_type if rtype != 18]))
            print("plausible_labels : {}".format(plausible_labels[idx]))


    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_concept_pair_ids = torch.tensor(total_concept_pair_ids, dtype=torch.long)
    total_relation_exist = torch.tensor(total_relation_exist, dtype=torch.long)
    total_relation_type = torch.tensor(total_relation_type, dtype=torch.long)
    total_plausible_label = torch.tensor(total_plausible_label, dtype=torch.long)
    dataset = TensorDataset(total_input_ids, total_attention_mask, total_concept_pair_ids, total_relation_exist, total_relation_type, total_plausible_label)
    return dataset


def convert_multidata2tensordataset(
        args, nlp, matcher, tokenizer,
        sentences, answers, plausible_labels,
        relation_exists, relation_types
):
    count = 0
    total_input_ids = []
    total_attention_mask = []
    total_concept_pair_idx = [] #cls 토큰같은거 생각안하고 토크나이즈 한거로 생각
    total_relation_exist = []
    total_relation_type = []
    total_plausible_label = []

    for sentence, answer, plausible_label in tqdm(zip(sentences, answers, plausible_labels), total=len(sentences), desc='convert multi task data'):
        sentence = standard_sentence(sentence)
        doc = nlp(sentence)
        matches = matcher(doc)

        processed_sentence = tokenizer(sentence)

        input_ids = processed_sentence['input_ids']
        attention_mask = processed_sentence['attention_mask']
        padding_input_ids = args.input_sentence_max_length - len(input_ids)
        input_ids += (padding_input_ids * [0])
        attention_mask += (padding_input_ids * [0])

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_plausible_label.append([plausible_label])


        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in sentence:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        sentence_span_concept = {}
        for match_id, start, end in matches:
            span = doc[start:end].text  # sentence의 span
            if span in sentence_span_concept.keys():
                continue
            start_position = char_to_word_offset[sentence.find(span)]
            end_position = char_to_word_offset[min(sentence.find(span) + len(span) - 1, len(char_to_word_offset) - 1)]

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

            if span not in sentence_span_concept:
                sentence_span_concept[span] = [tok_start_position, tok_end_position]
            # sentence_span_concept[span].append([tok_start_position, tok_end_position])

        concept_pair_to_idx = []
        for sentence_concept, start_end_idx in sentence_span_concept.items():
            sentence_tok_start, sentence_tok_end = start_end_idx
            concept_pair_to_idx.append([
                [sentence_tok_start, sentence_tok_end], sentence_span_concept[answer]
            ])
        total_concept_pair_idx.append(concept_pair_to_idx)

        for idx, pair_token_idx in enumerate(total_concept_pair_idx):
            padding = [[-1,-1], [-1, -1]]
            padding_list = []
            for i in range(args.max_length_pair_token-len(pair_token_idx)):
                padding_list.append(padding)
            total_concept_pair_idx[idx] = pair_token_idx + padding_list

        relation_exist_padding_idx = 2
        relation_type_padding_idx = 18
        for idx, (relation_exist_list, relation_type_list) in enumerate(zip(relation_exists, relation_types)):
            exist_padding = [relation_exist_padding_idx] * (args.max_length_exist-len(relation_exist_list))
            type_padding = [relation_type_padding_idx] * (args.max_length_type -len(relation_type_list))
            relation_exists[idx] = relation_exist_list+exist_padding
            relation_types[idx] = relation_type_list+type_padding
        total_relation_exist = relation_exists
        total_relation_type = relation_types
        count += 1
        if count == 4:
            break

    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_concept_pair_idx = torch.tensor(total_concept_pair_idx, dtype=torch.long)
    total_relation_exist = torch.tensor(total_relation_exist[:4], dtype=torch.long)
    total_relation_type = torch.tensor(total_relation_type[:4], dtype=torch.long)
    total_plausible_label = torch.tensor(total_plausible_label, dtype=torch.long)
    dataset = TensorDataset(
        total_input_ids,
        total_attention_mask,
        total_concept_pair_idx,
        total_relation_exist,
        total_relation_type,
        total_plausible_label
    )
    return dataset






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert dataset")
    parser.add_argument(
        "--path_to_train",
        type=str,
        default='../../../data/train/traindata.tsv'
    )
    parser.add_argument(
        '--path_to_training_labels',
        type=str,
        default='../../../data/train/trainlabels.tsv'
    )
    parser.add_argument(
        '--path_to_multitask_dataset_json',
        type=str,
        default='../../../data/multitask_dataset.json'
    )
    parser.add_argument(
        '--path_to_conceptnet',
        type=str,
        default='../../../conceptnet/assertions-570-en.csv'
    )
    parser.add_argument(
        '--path_to_lemma_json',
        type=str,
        default='../../../conceptnet/lemma_matching.json'
    )
    parser.add_argument(
        '--path_to_concept_word',
        type=str,
        default='../../../conceptnet/concept_word.txt'
    )

    args = parser.parse_args()
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'textcat'])
    matcher = load_matcher(args, nlp)
    conceptnet_vocab = read_concept_vocab(args)

    conceptnet_dataset = pd.read_csv(args.path_to_conceptnet, sep='\t', header=None,
                                     names=['relation', 'header', 'tail', 'weight'])
    conceptnet_relation = sorted(conceptnet_dataset['relation'].unique().tolist())
    conceptnet_relation.insert(0, 'no relation')
    relation2idx = {}
    idx2relation = {}
    for idx, relation in enumerate(conceptnet_relation):
        relation2idx[relation] = idx
        idx2relation[idx] = relation
    print(relation2idx, idx2relation)

    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    _, article_title, prev_sentences, now_sentences, next_sentences, filler_option = retrieve_all_instances_from_dataset(
        train_set)  # len : 19975개 각각
    training_label_set = pd.read_csv(args.path_to_training_labels, sep="\t", header=None, names=["Id", "Label"])
    plausible_label = retrieve_labels_from_dataset_for_classification(training_label_set)

    with open(args.path_to_multitask_dataset_json, 'r', encoding='utf8') as f:
        multi_dataset = json.load(f)
    sentences, sentences_concepts, answers, answers_concepts, relation_exists, relation_types, concept_pairs\
        = read_my_multi_dataset(multi_dataset, relation2idx)

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    total_concept_pair_to_idx = []
    for sentence, answer, concept_pair in tqdm(zip(sentences[:10], answers[:10], concept_pairs[:10]), total=len(sentences[:10]), desc='token start end belong to span'):
        sentence = standard_sentence(sentence)
        doc = nlp(sentence)
        matches = matcher(doc)

        sentence_tokenize = " ".join(tokenizer.tokenize(sentence))
        input_ids = tokenizer(sentence)['input_ids']
        #electra cls 101 / sep 102
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in sentence:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] +=c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens)-1)

        sentence_span_concept = {}
        for match_id, start, end in matches:
            span = doc[start:end].text #sentence의 span
            if span in sentence_span_concept.keys():
                continue
            start_position = char_to_word_offset[sentence.find(span)]
            end_position = char_to_word_offset[min(sentence.find(span)+len(span)-1, len(char_to_word_offset)-1)]

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
            if end_position < len(doc_tokens)-1:
                tok_end_position = orig_to_tok_index[end_position + 1 ] - 1
            else:
                tok_end_position = len(all_doc_tokens)-1

            if span not in sentence_span_concept:
                sentence_span_concept[span] = [tok_start_position, tok_end_position]
            # sentence_span_concept[span].append([tok_start_position, tok_end_position])
        concept_pair_to_idx = []
        for sentence_concept, start_end_idx in sentence_span_concept.items():
            sentence_tok_start, sentence_tok_end = start_end_idx
            concept_pair_to_idx.append([
                [sentence_tok_start, sentence_tok_end], sentence_span_concept[answer]
            ])
        total_concept_pair_to_idx.append(concept_pair_to_idx)
        '''
        for pair in concept_pair:
            head, tail = pair[0], pair[1]
            temp = sentence_span_concept[head.replace("_", " ")]
            head_tok_start, head_tok_end = sentence_span_concept[head.replace("_", " ")][0]
            tail_tok_start, tail_tok_end = sentence_span_concept[tail.replace("_", " ")][0]
            concept_pair_to_idx.append([
                [head_tok_start, head_tok_end], [tail_tok_start, tail_tok_end]
            ])
        '''
    max_length_pair_token_idx = 0
    for pair_token_idx in total_concept_pair_to_idx:
        if max_length_pair_token_idx < len(pair_token_idx):
            max_length_pair_token_idx = len(pair_token_idx)
    for idx, pair_token_idx in enumerate(total_concept_pair_to_idx):
        padding = [[-1, -1], [-1, -1]]
        padding_list = []
        for i in range(max_length_pair_token_idx - len(pair_token_idx)):
            padding_list.append(padding)

        total_concept_pair_to_idx[idx] = pair_token_idx + padding_list
    total_pair_token_idx = torch.tensor(total_concept_pair_to_idx, dtype=torch.long)
    max_len_relation_exist = 0
    max_len_relation_type = 0
    relation_exist_padding_idx = 2  # 추후에 arg로 넣는방법
    relation_type_padding_idx = 18  # 추후에 arg로 넣자
    for relation_exist_list, relation_type_list in tqdm(zip(relation_exists, relation_types), total=len(relation_exists), desc='find max length related to relation'):
        if max_len_relation_exist < len(relation_exist_list):
            max_len_relation_exist = len(relation_exist_list)
        if max_len_relation_type < len(relation_type_list):
            max_len_relation_type = len(relation_type_list)
    for idx, (relation_exist_list, relation_type_list) in tqdm(enumerate(zip(relation_exists, relation_types)), total=len(relation_exists), desc='make relation tensor'):
        exist_padding = [relation_exist_padding_idx] * (max_len_relation_exist-len(relation_exist_list))
        type_padding = [relation_type_padding_idx] * (max_len_relation_type-len(relation_type_list))

        relation_exists[idx] = relation_exist_list + exist_padding
        relation_types[idx] = relation_type_list + type_padding

    total_relation_exist = torch.tensor(relation_exists, dtype=torch.long)
    total_relation_type = torch.tensor(relation_types, dtype=torch.long)






