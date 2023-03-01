from itertools import product, combinations, permutations
from transformers import RobertaTokenizerFast
import pandas as pd
from tqdm import tqdm
from src.multi_task.func.my_utils import retrieve_all_instances_from_dataset, retrieve_labels_from_dataset_for_classification, retrieve_multitask_label_for_classification
import argparse
import re
import torch
from torch.utils.data import TensorDataset

def remove_special_token_ids(contain_special_token_list):
    return [token for token in contain_special_token_list if (token is not 0) and (token is not 2)]


def remove_trash(current_sentence):
    current_sentence = current_sentence.replace("(...)", "").strip().lower()

    return current_sentence


def get_relation_exist_and_type_label(pair_title, pair_content, knowledge_source):
    temp_knowledge = knowledge_source[
        (knowledge_source['score']>=0.5) &
        ((knowledge_source['header'].isin(pair_title)) & (knowledge_source['tail'].isin(pair_content)))
    ]
    pair_title = [title for title in pair_title if title in temp_knowledge['header'].tolist()]
    pair_content = [content for content in pair_content if content in temp_knowledge['tail'].tolist()]
    total_pair = []
    relation_exist_label = []
    relation_type_label = []
    pairs = [pair_title, pair_content]
    pairs = list(product(*pairs))
    for pair in pairs:
        title, content = pair
        total_pair.append(pair)
        temp_temp_knowledge = temp_knowledge[(temp_knowledge['header'] == title) & (temp_knowledge['tail'] == content)]
        if not temp_temp_knowledge.empty:
            relation_exist_label.append(temp_temp_knowledge['score'].tolist())
            relation_type_label.append(temp_temp_knowledge['relation'].tolist())
        else:
            relation_exist_label.append(0)
            relation_type_label.append(0)
    return relation_exist_label, relation_type_label, total_pair


# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
#
# token_title = [0, 6179, 7, 10391, 7, 2486, 16623, 1936, 2667, 166, 8774, 2]
# token_content = [3226, 38551, 17515, 5, 21487, 4846, 1437, 1437, 1009, 7627, 30450, 7, 10310, 370, 38551, 17515, 2, 3226, 38551, 17515, 9612, 19, 2710, 2, 2]
#
#
# token_title = remove_special_token_ids(token_title)
# token_content = remove_special_token_ids(token_content)
#
#
# # 타이틀과 콘텐트의 pair 리스트
# title = [clear[1:] if 'Ġ' in clear else clear for clear in tokenizer.convert_ids_to_tokens(token_title)]
# content = [clear[1:] if 'Ġ' in clear else clear for clear in tokenizer.convert_ids_to_tokens(token_content)]
# products = [title, content]
# products = list(product(*products))
#
# product_list = []
#
# transomcs = pd.read_csv('../../../data/transomcs/TransOMCS_full.txt',
#                         sep='\t', header=None, names=['header', 'relation', 'tail', 'score'])
# # 스코어 threshold 나중에 상수로
# temp_transomcs = transomcs[(transomcs['score']>=0.5) & ((transomcs['header'].isin(title)) | (transomcs['tail'].isin(content)))]
# print(temp_transomcs)
#
# relation_exist = []
#
# for prod in tqdm(products, desc='find relation_exist'):
#     sub, obj = prod
#     condition = (temp_transomcs['header'] == sub) & (temp_transomcs['tail'] == obj)
#     if not temp_transomcs[condition].empty:
#         print(temp_transomcs[condition])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataset analysis")
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
        '--path_to_knowledge',
        type=str,
        default='../../../data/transomcs/TransOMCS_full.txt'
    )
    parser.add_argument(
        '--path_to_multitask_label',
        type=str,
        default='../../../data/transomcs/multitask_label.csv'
    )

    args = parser.parse_args()

    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # load dataset
    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    training_label_set = pd.read_csv(args.path_to_training_labels, sep="\t", header=None, names=["Id", "Label"])
    knowledge_source = pd.read_csv(args.path_to_knowledge, sep='\t', header=None, names=['header', 'relation', 'tail', 'score']) #transomcs KB
    multi_task_label = pd.read_csv(args.path_to_multitask_label, sep='\t').drop(["Unnamed: 0"], axis=1) #관계 존재 라벨이랑 유형 라벨

    # process data
    _, article_titles, prev_sentences, now_sentences, next_sentences, filler_options = retrieve_all_instances_from_dataset(train_set) #len : 19975개 각각
    plausible_labels = retrieve_labels_from_dataset_for_classification(training_label_set)
    exist_labels, relation_types_labels, pairs = retrieve_multitask_label_for_classification(multi_task_label)

    # total data -> tensordataset
    total_relation_exist, total_relation_type, total_pair = [], [], []
    total_input_ids, total_attention_mask, total_plausible_label = [], [], []

    # tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')


    for index, (article_title, prev_sentence, now_sentence, next_sentence, option, plausible_label, exist_label, relation_type_label, pair) in tqdm(
        enumerate(zip(article_titles, prev_sentences, now_sentences, next_sentences, filler_options, plausible_labels, exist_labels, relation_types_labels, pairs)),
        desc='make relation exist label, type lable', total=len(article_titles)
    ):
        article_title = remove_trash(article_title)
        prev_sentence = remove_trash(prev_sentence)
        now_sentence = remove_trash(now_sentence)
        next_sentence = remove_trash(next_sentence)
        content_sentence = prev_sentence+now_sentence+next_sentence

        process_article_title = tokenizer(article_title)
        process_prev_sentence = tokenizer(prev_sentence)
        process_now_sentence = tokenizer(now_sentence)
        process_next_sentence = tokenizer(next_sentence)
        process_content = tokenizer(content_sentence)

        tokens_article_title = process_article_title.encodings[0].tokens
        tokens_content = process_content.encodings[0].tokens

        pair_title = [clear[1:] if 'Ġ' in clear else clear for clear in tokens_article_title[1:-1]]
        pair_content = [clear[1:] if 'Ġ' in clear else clear for clear in tokens_content[1:-1]]

        # 한 줄 단위
        input_ids = process_article_title['input_ids'] + process_content['input_ids'][1:]
        attention_mask = process_article_title['attention_mask'] + process_content['attention_mask'][1:]
        input_ids_padding = [1] * (300-len(input_ids))
        attention_mask_padding = [0] * (300-len(attention_mask))
        input_ids += input_ids_padding
        attention_mask += attention_mask_padding

        # total
        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_plausible_label.append(plausible_label)

        exist_instance = []
        type_instance = []
        instance_max_length = 1500
        for exist in exist_label:
            if type(exist) == list:
                exist_instance.append(exist[0])
            else:
                exist_instance.append(0)
        for relation_type in relation_type_label:
            if type(relation_type) == list:
                type_instance.append(relation_type[0])
            else:
                type_instance.append(0)
        exist_instance += [-1] * (instance_max_length - len(exist_instance))
        type_instance += [-1] * (instance_max_length - len(type_instance))
        first_index = [idx for idx, no_zero in enumerate(exist_instance) if not no_zero == 0][0]
        head, tail = pair[first_index]
        title_tokens = process_article_title.encodings[0].tokens
        title_offset = process_article_title.encodings[0].offsets
        for tokens, offset in zip(title_tokens, title_offset):
            if tokens == head or tokens == 'Ġ'+head:
                print(offset)
        total_relation_exist.append(exist_instance)
        total_relation_type.append(type_instance)
        if index < 2:
            print()
            print("*** Example ***")
            print("title : {}".format(article_title))
            print("token title : {}".format(tokens_article_title))
            print("input ids title : {}".format(process_article_title['input_ids']))
            print("content : {}".format(content_sentence))
            print("token content : {}".format(tokens_content))
            print("input ids content : {}".format(process_content['input_ids']))
            print("relation exist label : {}".format(exist_instance))
            print("relation type label : {}".format(type_instance))
            print()


        #relation_exist_label, relation_type_label, instance_pair = get_relation_exist_and_type_label(pair_title, pair_content, knowledge_source)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_plausible_label = torch.tensor(total_plausible_label, dtype=torch.long)
    total_relation_exist = torch.tensor(total_relation_exist, dtype=torch.float)
    total_relation_type = torch.tensor(total_relation_type, dtype=torch.long)
    dataset = TensorDataset(total_input_ids, total_attention_mask, total_plausible_label, total_relation_exist, total_relation_type)
    print(dataset)
    # total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    # total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    # total_labels = torch.tensor(plausible_labels, dtype=torch.long)
    # total_relation_exist_label = torch.tensor(total_relation_exist_label, dtype=torch.float)
    # total_relation_type_label = torch.tensor(total_relation_type_label, dtype=torch.long)
    # dataset = TensorDataset(total_input_ids, total_attention_mask, total_labels, total_relation_exist_label, total_relation_type_label)
    # return dataset

    # save_pd = pd.DataFrame({
    #     'exist_label': total_relation_exist,
    #     'relation_type': total_relation_type,
    #     'pair': total_pair
    # })
    '''data frame
        col    exist label          | relation_type                 | pair
    row        [0, 0, 0, [0.85], ..]| [0, 0, 0, ['receives ...']]   |('to', 'loss'), ('to', 'clean'), ...]
    '''
    #print(save_pd)

    #save_pd.to_csv('../../../data/transomcs/multitask_label.csv', sep='\t')
