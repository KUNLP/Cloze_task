import torch
from torch.utils.data import TensorDataset
from transformers import RobertaTokenizerFast, RobertaTokenizer
import pandas as pd

from typing import List, Tuple, Dict
import argparse
import logging
from tqdm import tqdm
from itertools import product
import ast

logging.basicConfig(level=logging.DEBUG)

def remove_skip(current_sentence):
    return current_sentence.replace("(...)", "").strip()


def remove_special_token_ids(contrain_special_token_list):
    return [token for token in contrain_special_token_list if (token is not 0) and (token is not 2)]


def get_relation_exist_and_type_label(pair_title, pair_content, knowledge_source):
    temp_knowledge = knowledge_source[
        (knowledge_source['score']>=0.5) &
        ((knowledge_source['header'].isin(pair_title)) | (knowledge_source['tail'].isin(pair_content)))
    ]

    relation_exist_label = []
    relation_type_label = []
    pairs = [pair_title, pair_content]
    pairs = list(product(*pairs))
    for pair in pairs:
        title, content = pair
        condition = (temp_knowledge['header'] == title) & (temp_knowledge['tail'] == content)
        if not temp_knowledge[condition].empty:
            relation_exist_label.append(temp_knowledge[condition]['score'])
            relation_type_label.append(temp_knowledge[condition]['relation'])
        else:
            relation_exist_label.append(0)
            relation_type_label.append(0)
    return relation_exist_label, relation_type_label


def write_predictions_to_file(
    path_to_predictions: str, ids: List[str], predictions: List, subtask: str
) -> pd.DataFrame:

    if subtask == "classification":
        predictions = convert_class_indices_to_labels(predictions)

    dataframe = pd.DataFrame({"Id": ids, "Label": predictions})
    logging.info(f"--> Writing predictions to {path_to_predictions}")
    dataframe.to_csv(path_to_predictions, sep="\t", index=False, header=False)

    return dataframe


def convert_class_indices_to_labels(class_indices: List[int]) -> List[str]:
    """Convert integer class indices to str labels.

    :param class_indices: list of int class indices (0 to 2)
    :return: list of label strs from set "IMPLAUSIBLE" / "NEUTRAL" / "PLAUSIBLE"
    """
    labels = ["IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"]
    return [labels[class_index] for class_index in class_indices]


def retrieve_multitask_label_for_classification(
    multi_label: pd.DataFrame
) -> Tuple[List[List[int]], List[List[int]], List[Tuple[str]]]:

    # transomcs relation_type
    standard_type = {
        'CapableOf': 1,
        'UsedFor': 2,
        'HasProperty': 3,
        'AtLocation': 4,
        'HasA': 5,
        'ReceivesAction': 6,
        'InstanceOf': 7,
        'PartOf': 8,
        'CausesDesire': 9,
        'MadeOf': 10,
        'CreatedBy': 11,
        'Causes': 12,
        'HasPrerequisite': 13,
        'HasSubevent': 14,
        'MotivatedByGoal': 15,
        'HasLastSubevent': 16,
        'Desires': 17,
        'HasFirstSubevent': 18,
        'DefinedAs': 19,
        'LocatedNear': 20
    }

    exist_label, relation_type, pair = [], [], []
    for _, row in multi_label.iterrows():
        exist_label.append(ast.literal_eval(row['exist_label']))
        relation_temp = [content for content in ast.literal_eval(row['relation_type'])]
        relation = []
        for content in relation_temp:
            if type(content) == int:
                relation.append(content)
            else:
                relation.append([standard_type[rel] for rel in content])
        relation_type.append(relation)
        pair.append(ast.literal_eval(row['pair']))

    # exist label = [0, 0, 0, [0.85, 0.9], 0 ...]
    # relation type = [0, 0, 0, [1,9], 0 ...]
    # pair = [('to', 'lose'), ('to', 'weight'), ...]
    return exist_label, relation_type, pair


def retrieve_labels_from_dataset_for_classification(
    label_set: pd.DataFrame,
) -> List[int]:
    """Retrieve labels from dataset.

    :param label_set: dataframe with class labels
    :return: list of int class labels 0, 1 or 2 (IMPLAUSIBLE, NEUTRAL, PLAUSIBLE)
    """
    # the labels are already in the right order for the training instances, so we can just put them in a list
    label_strs = list(label_set["Label"])
    label_ints = []
    for label_str in label_strs:
        if label_str == "IMPLAUSIBLE":
            label_ints.append(0)
        elif label_str == "NEUTRAL":
            label_ints.append(1)
        elif label_str == "PLAUSIBLE":
            label_ints.append(2)
        else:
            raise ValueError(f"Label {label_str} is not a valid plausibility class.")
    return label_ints


def retrieve_instances_from_dataset(
    dataset: pd.DataFrame,
) -> Tuple[List[str], List[str]]:

    dataset = dataset.fillna("")

    ids = []
    instances = []
    for _, row in dataset.iterrows():
        for filler_index in range(1, 6):
            ids.append(f"{row['Id']}_{filler_index}")

            sent_with_filler = row["Sentence"].replace(
                "______", row[f"Filler{filler_index}"]
            )
            instances.append(sent_with_filler)
    return ids, instances


def retrieve_all_instances_from_dataset(dataset):
    '''
    데이터프레임에서 빈칸이 채워진 현재문장과 이전 문장 다음 문장을 반환
    '''
    dataset = dataset.fillna("")

    ids = []
    article_title = []
    prev_instances = []
    now_instances = []
    next_instances = []
    filler_option = []

    for _, row in dataset.iterrows():
        for filler_index in range(1,6):
            ids.append(f"{row['Id']}_{filler_index}")
            article_title.append(row["Article title"])
            sent_with_filler = row["Sentence"].replace(
                "______", row[f"Filler{filler_index}"]
            )

            now_instances.append(sent_with_filler)
            prev_instances.append(row["Previous context"])
            next_instances.append(row["Follow-up context"])
            filler_option.append(row[f"Filler{filler_index}"])
    return ids, article_title, prev_instances, now_instances, next_instances, filler_option


# 여기서
def convert_data2tensordataset_for_multitask(
        article_titles,
        prev_sentences, now_sentences, next_sentences, filler_option,
        plausible_labels, tokenizer, max_length
    ):
    total_input_ids, total_attention_mask, total_relation_exist_label, total_relation_type_label = [], [], [], []
    knowledge_source = pd.read_csv('../../../data/transomcs/TransOMCS_full.txt', sep='\t', header=None,
                                   names=['header', 'relation', 'tail', 'score'])

    # <s> 토큰 index 0
    # </s> 토큰 index 2
    for index, (article_title, prev_sentence, now_sentence, next_sentence, option) in tqdm(
        enumerate(zip(article_titles, prev_sentences, now_sentences, next_sentences, filler_option)),
        desc='convert data to tensordataset',
        total=len(article_titles)
    ):

        article_title = remove_skip(article_title)
        prev_sentence = remove_skip(prev_sentence)
        now_sentence = remove_skip(now_sentence)
        next_sentence = remove_skip(next_sentence)
        content_sentence = prev_sentence+now_sentence+next_sentence

        # input_ids랑 attention_mask랑 encodings
        process_article_title = tokenizer(article_title)
        process_prev_sentence = tokenizer(prev_sentence)
        process_now_sentence = tokenizer(now_sentence)
        process_next_sentence = tokenizer(next_sentence)
        process_content = tokenizer(content_sentence)

        # title
        input_ids_article_title = process_article_title['input_ids']
        attention_mask_article_title = process_article_title['attention_mask']
        offset_article_title = process_article_title.encodings[0].offsets
        tokens_article_title = process_article_title.encodings[0].tokens

        # content
        input_ids_content = process_content['input_ids']
        attention_mask_content = process_content['attention_mask']
        offset_article_title = process_content.encodings[0].offsets
        tokens_content = process_content.encodings[0].tokens

        pair_title = [clear[1:] if 'Ġ' in clear else clear for clear in tokens_article_title[1:-1]]
        pair_content = [clear[1:] if 'Ġ' in clear else clear for clear in tokens_content[1:-1]]

        relation_exist_label, relation_type_label = get_relation_exist_and_type_label(pair_title, pair_content, knowledge_source)

        input_ids = input_ids_article_title + input_ids_content[1:]
        attention_mask = attention_mask_article_title + attention_mask_content[1:]
        assert len(input_ids) == len(attention_mask)

        input_ids_padding = [1] * (max_length-len(input_ids))
        attention_mask_padding = [0] * (max_length-len(attention_mask))

        input_ids += input_ids_padding
        attention_mask += attention_mask_padding


        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_relation_exist_label.append(relation_exist_label)
        total_relation_type_label.append(relation_type_label)

        if index < 2:
            print()
            print("*** Example ***")
            print("title : {}".format(article_title))
            print("token title : {}".format(tokens_article_title))
            print("input ids title : {}".format(input_ids_article_title))
            print("content : {}".format(content_sentence))
            print("token content : {}".format(tokens_content))
            print("input ids content : {}".format(input_ids_content))
            #print("all token : {}".format(" ".join([str(x) for x in total_input_ids[-1]])))
            #print("attention mask : {}".format(" ".join([str(x) for x in total_attention_mask[-1]])))
            print()
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_labels = torch.tensor(plausible_labels, dtype=torch.long)
    total_relation_exist_label = torch.tensor(total_relation_exist_label, dtype=torch.float)
    total_relation_type_label = torch.tensor(total_relation_type_label, dtype=torch.long)
    dataset = TensorDataset(total_input_ids, total_attention_mask, total_labels, total_relation_exist_label, total_relation_type_label)

    #필요한 리턴
    #input_ids / attention_mask / plausible labels / 제목 토큰 인덱스와 내용문장 토큰 인덱스의 페어쌍
    #19995개로 다 일케 나와야?
    return dataset



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

    args = parser.parse_args()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)

    _, article_title, prev_sentences, now_sentences, next_sentences, filler_option = retrieve_all_instances_from_dataset(train_set) #len : 19975개 각각
    training_label_set = pd.read_csv(args.path_to_training_labels, sep="\t", header=None, names=["Id", "Label"])


    #이게 토탈 라벨리스트랑 같음
    plausible_label = retrieve_labels_from_dataset_for_classification(training_label_set)



    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

    dataset = convert_data2tensordataset_for_multitask(
        article_title, prev_sentences, now_sentences, next_sentences, filler_option,
        plausible_label, tokenizer, 300
    )
    print(dataset)