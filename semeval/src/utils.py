from data import retrieve_instances_from_dataset
from format_checker_for_dataset import check_format_of_dataset


from typing import List, Tuple, Dict
import pandas as pd
import argparse
import logging
import re
from transformers import ElectraTokenizerFast
#import nltk
from tqdm import tqdm
from data import retrieve_labels_from_dataset_for_classification
import torch
from torch.utils.data import TensorDataset


logging.basicConfig(level=logging.DEBUG)

def retrieve_all_instances_from_dataset(
        dataset: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    '''
    데이터프레임에서 빈칸이 채워진 현재문장과 이전 문장 다음 문장을 반환
    '''
    dataset = dataset.fillna("")

    ids = []
    prev_instances = []
    now_instances = []
    next_instances = []


    for _, row in dataset.iterrows():
        for filler_index in range(1,6):
            ids.append(f"{row['Id']}_{filler_index}")

            sent_with_filler = row["Sentence"].replace(
                "______", row[f"Filler{filler_index}"]
            )

            now_instances.append(sent_with_filler)
            prev_instances.append(row["Previous context"])
            next_instances.append(row["Follow-up context"])

    return ids, prev_instances, now_instances, next_instances


def retrieve_option_morph_from_dataset(
        dataset: pd.DataFrame,
) -> Tuple[List[str]]:
    '''
    보기에 있는 단어들을 현재 빈칸에 넣어서 형태소 분석을 통하고 형태소를 비교하고자 하였다.
    ['noun', 'noun', 'noun', 'noun', 'noun'] 이런식으로 나오는데 아직 명사 구 형태로 나오는건 안됨. 수정 필요
    '''
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('punkt')
    dataset = dataset.fillna("")
    ids = []
    options = []
    for _, row in tqdm(dataset.iterrows(), desc='read dataset'):
        filler_list = []

        for filler_index in range(1,6):

            ids.append(f"{row['Id']}_{filler_index}")
            sent_with_filler = row["Sentence"].replace(
                "______", row[f"Filler{filler_index}"]
            )

            sent = nltk.word_tokenize(sent_with_filler)

            sent_with_filler = nltk.pos_tag(sent)
            # [('*', 'NNS'), ('Lose', 'NNP'), ('Weight', 'NNP'), ('Fast', 'NNP'), ('with', 'IN'), ('stress', 'NN')]

            # 이 부분 수정할것
            #이렇게 하면 the circumference, your rectangle 과 같이 구로 된 거 해석이 안된다.
            for content in sent_with_filler:
                if content[0] == row[f"Filler{filler_index}"]:
                    filler_list.append({content[0]: content[1]})
        options.append(filler_list)
    return options



def dict_values_to_value(
        dictionary: Dict,
) -> str:
    '''
    dict의 값을 일반적인 문자열로 반환
    '''
    value = list(dictionary.values())[0]
    return value


def get_static_option(
        options: List[List[str]],
) -> None:
    '''
    보기가 전부 같은 형태소인지 보려고 했었음. 아직 제대로 구현 안됌
    '''
    all_same = 0
    not_all_same = 0
    for content in tqdm(options, desc='get static option'):
        content_morph_dict = {}
        for index in range(5):
            content_morph_dict[index] = dict_values_to_value(content[index])

        if content_morph_dict[0] == content_morph_dict[1] == content_morph_dict[2] == content_morph_dict[3] == content_morph_dict[4]:
            all_same += 1
        else:
            not_all_same += 1
            print(content)
    print('all same option morph : {}, not all : {}'.format(all_same, not_all_same))


def remove_skip(
        current_sentence : str
) -> str:
    '''
    데이터에서 (...)는 의미없는 토큰으로 되므로 그냥 제거 replace "(...)" -> ""
    좌우 공백 제거 strip
    '''
    return current_sentence.replace("(...)", "").strip()



def get_all_sentence_length(
        prev_sentences : List[str],
        now_sentences : List[str],
        next_sentences : List[str]
) -> int:
    '''
    길이 통계 내보려고 했던거. 이전, 현재, 이후 문자 토큰한거 길이재서
    최대 길이 254로 최대 길이는 256으로 잡아도 될듯하다

    '''
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    max_length = 0
    for prev, now, next in zip(prev_sentences, now_sentences, next_sentences):
        prev = remove_skip(prev)
        now = remove_skip(now)
        next = remove_skip(next)
        length = len(tokenizer.tokenize(prev)+tokenizer.tokenize(now)+tokenizer.tokenize(next))
        if length > max_length:
            max_length = length
    print(max_length)
    return max_length


def convert_data2tensordataset(prev_sentences, now_sentences, next_sentences, label_list, tokenizer, max_length):
    total_input_ids, total_attention_mask = [], []
    for index, data in enumerate(
            tqdm(zip(prev_sentences, now_sentences, next_sentences), desc="convert data to tensordataset")):
        all_sentence = remove_skip(data[0]) + remove_skip(data[1]) + remove_skip(data[2])
        token_all_sentence = tokenizer.tokenize(all_sentence)
        input_ids = tokenizer(all_sentence)['input_ids']
        attention_mask = [1] * len(input_ids)
        input_ids_padding = [1] * (max_length - len(input_ids))
        assert len(input_ids) == len(attention_mask)

        attention_mask_padding = [0] * (max_length - len(attention_mask))

        input_ids += input_ids_padding
        attention_mask += attention_mask_padding

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)

        if index < 2:
            print()
            print("*** Example ***")
            print("all_sentence : {}".format(all_sentence))
            print("token_all_sentence: {}".format(" ".join([str(x) for x in token_all_sentence])))
            print("input_ids: {}".format(" ".join([str(x) for x in total_input_ids[-1]])))
            print("attention_mask: {}".format(" ".join([str(x) for x in total_attention_mask[-1]])))
            print()

    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.float)
    total_labels = torch.tensor(label_list, dtype=torch.long)
    dataset = TensorDataset(total_input_ids, total_attention_mask, total_labels)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataset analysis")
    parser.add_argument(
        "--path_to_train",
        type=str,
        default='../data/train/traindata.tsv'
    )
    parser.add_argument(
        '--path_to_training_labels',
        type=str,
        default='../data/train/trainlabels.tsv'
    )
    args = parser.parse_args()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)

    # 길이 관련 통계 보려고
    #_, prev_sentences, now_sentences, next_sentences = retrieve_all_instances_from_dataset(train_set)
    #get_all_sentence_length(prev_sentences, now_sentences, next_sentences)

    # 보기의 형태소가 어떤지 보려고
    #options_morph = retrieve_option_morph_from_dataset(train_set)
    #get_static_option(options_morph)

    _, prev_sentences, now_sentences, next_sentences = retrieve_all_instances_from_dataset(train_set)
    training_label_set = pd.read_csv(args.path_to_training_labels, sep="\t", header=None, names=["Id", "Label"])


    #이게 토탈 라벨리스트랑 같음
    label_list = retrieve_labels_from_dataset_for_classification(training_label_set)

    # for sen in now_sentences:
    #     if sen == None:
    #         print('empty!')
    # none_count = 0
    # blank_count = 0
    # null_count = 0
    # space_count = 0
    # for index, sen in enumerate(prev_sentences):
    #     if sen == None:
    #         none_count +=1
    #         print('None')
    #     elif sen == '':
    #         blank_count +=1
    #         print('idx: {}, blank: {}'.format(index, sen))
    #     elif sen == 'null':
    #         null_count += 1
    #         print('null')
    #     elif sen == ' ':
    #         space_count +=1
    #         print('idx: {}, space : {}'.format(index, sen))
    # print('none count : {}, blank count : {}, null count : {}, space count : {}'.format(none_count, blank_count, null_count, space_count))