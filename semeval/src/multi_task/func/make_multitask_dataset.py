import argparse, re, string, json
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product

import spacy, nltk
from spacy.matcher import Matcher

from src.multi_task.func.my_utils import retrieve_all_instances_from_dataset, retrieve_labels_from_dataset_for_classification
from src.multi_task.func.grounding_concept import read_concept_vocab, load_matcher, lemmatize, ground_mention_concepts, hard_ground
from src.multi_task.func.multitask_utils_final import standard_sentence


def retrieve_context_filler_list_from_dataset(dataset):
    '''
    분장 합쳐진거 하나랑 정답 리스트 다섯개 한번에 볼 수 있게
    '''
    dataset = dataset.fillna("")

    ids = []
    article_title = []
    prev_instances = []
    now_instances = []
    next_instances = []
    filler_option = []

    for _, row in dataset.iterrows():
        for filler_index in range(1, 6):
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


def retrieve_pairs_relation_exist_and_relation_type_from_conceptnet(conceptnet_dataset, context_concepts, answer_concepts):
    temp_conceptnet = conceptnet_dataset[(conceptnet_dataset['header'].isin(context_concepts))|(conceptnet_dataset['tail'].isin(answer_concepts))]
    pairs = [context_concepts, answer_concepts]
    pairs = list(product(*pairs))

    relation_exist = []
    relation_type = []
    for pair in pairs:
        context_concept, answer_concept = pair
        condition = (temp_conceptnet['header'] == context_concept) & (temp_conceptnet['tail'] == answer_concept)
        if not temp_conceptnet[condition].empty:
            relation_exist.append(1)
            relation_type.append(temp_conceptnet[condition]['relation'].values[0])
        else:
            relation_exist.append(0)
            relation_type.append(0)
    return pairs, relation_exist, relation_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make multitask label tsv")
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
        "--path_to_conceptnet_csv_en",
        type=str,
        default='../../../conceptnet/assertions-570-en.csv'
    )
    parser.add_argument(
        '--path_to_concept_word',
        type=str,
        default='../../../conceptnet/concept_word.txt'
    )
    parser.add_argument(
        "--path_to_lemma_json",
        type=str,
        default='../../../conceptnet/lemma_matching.json'
    )
    parser.add_argument(
        '--path_to_multitask_dataset',
        type=str,
        default='../../../data/multitask_dataset_train.json'
    )
    parser.add_argument('--path_to_test_data', type=str, default='../../../data/test/testdata.tsv')
    parser.add_argument('--path_to_save_test_data', type=str, default='../../../data/multitask_dataset_test.json')

    args = parser.parse_args()

    concept_vocab = read_concept_vocab(args)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

    matcher = load_matcher(args, nlp)

    #load dataset
    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    training_label_set = pd.read_csv(args.path_to_training_labels, sep="\t", header=None, names=["Id", "Label"])
    conceptnet_dataset = pd.read_csv(args.path_to_conceptnet_csv_en, sep='\t', header=None, names=["relation", "header", "tail", "weight"])

    test_set = pd.read_csv(args.path_to_test_data, sep='\t', quoting=3)


    #process data
    #idx, article_titles, prev_sentences, now_sentences, next_sentences, filler_options = retrieve_all_instances_from_dataset(train_set[:10]) #len : 19975개 각각

    # _, article_titles, prev_instances, now_instances, next_instances, answer_instances = \
    #     retrieve_context_filler_list_from_dataset(train_set)
    # plausible_labels = retrieve_labels_from_dataset_for_classification(training_label_set)

    _, article_titles, prev_instances, now_instances, next_instances, answer_instances = \
        retrieve_context_filler_list_from_dataset(test_set)



    dataset = []
    for idx, (prev_instance, now_instance, next_instance, answer_instance) in tqdm(enumerate(zip(prev_instances, now_instances, next_instances, answer_instances)),
                                     desc="make test dataset json", total=len(article_titles)):
        content = {}
        sentence = standard_sentence(prev_instance + now_instance + next_instance)
        doc = nlp(sentence)
        matches = matcher(doc)

        sentence_span_concept = []
        for match_id, start, end in matches:
            span = doc[start:end].text
            if span in sentence_span_concept:
                continue
            else:
                sentence_span_concept.append(span)

        doc = nlp(answer_instance)
        matches = matcher(doc)
        answer_span_concept = []
        for match_id, start, end in matches:
            span = doc[start:end].text
            if span in answer_span_concept:
                continue
            else:
                answer_span_concept.append(span)
        content['prev_instance'] = prev_instance
        content['now_instance'] = now_instance
        content['next_instance'] = next_instance
        content['answer_instance'] = answer_instance
        content['sentence_span_concept'] = sentence_span_concept
        content['answer_span_concept'] = answer_span_concept
        dataset.append(content)
        # if idx == 10:
        #     break


        # # 문장의 컨셉들
        # context_concepts = list(ground_mention_concepts(args, nlp, matcher, context))
        #
        # # 정답의 컨셉 (구 형태라도 컨셉만 뽑아서)
        # answer_dict = {}
        #
        # concept_pairs = {}
        # relation_exist = {}
        # relation_type = {}
        # for filler in filler_list:
        #     answer_concepts = list(ground_mention_concepts(args, nlp, matcher, filler))
        #     answer_dict[filler] = answer_concepts
        #
        #     #헤더와 테일 바꾼건 아직 안했다. 성능안나오면 나중에 해볼것
        #
        #     pairs, relation_exist_pairs, relation_type_pairs = retrieve_pairs_relation_exist_and_relation_type_from_conceptnet(conceptnet_dataset, context_concepts, answer_concepts)
        #     concept_pairs[filler] = pairs
        #     relation_exist[filler] = relation_exist_pairs
        #     relation_type[filler] = relation_type_pairs
        #
        # content["context"] = context
        # content["answers"] = filler_list
        # content["context_concepts"] = context_concepts
        # content["answers_concepts"] = answer_dict
        # content["concept_pairs"] = concept_pairs
        # content["relation_exist"] = relation_exist
        # content["relation_type"] = relation_type
        # dataset.append(content)

    with open(args.path_to_save_test_data, 'w', encoding='utf8') as f:
        json.dump(dataset, f, indent=4)










