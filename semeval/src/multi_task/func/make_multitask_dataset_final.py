import argparse, re, json
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
import spacy

from src.multi_task.func.my_utils import retrieve_labels_from_dataset_for_classification
from src.multi_task.func.grounding_concept import read_concept_vocab, load_matcher
from src.multi_task.func.multitask_utils_final import standard_sentence


def retrieve_all_processed_data_from_dataset(dataset):
    dataset = dataset.fillna("")

    ids = []
    titles = []
    prev_instances = []
    now_instances = []
    next_instances = []
    processed_sentence_instances = [] #standard하고 빈칸에 정답 채워넣은 문장
    answer_instances = []
    original_answer_char_positions = []

    for _, row in dataset.iterrows():
        for answer in range(1,6):
            ids.append(f"{row['Id']}_{answer}")
            titles.append(row['Article title'])

            sentence = row['Previous context'] + row['Sentence'] + row['Follow-up context']
            sentence = standard_sentence(sentence)
            original_answer_char_positions.append(sentence.find("______"))
            answer_instances.append(row[f"Filler{answer}"])
            prev_instances.append(row["Previous context"])
            next_instances.append(row["Follow-up context"])
            now_instances.append(row["Sentence"].replace("______", row[f"Filler{answer}"]))
            processed_sentence_instances.append(sentence.replace("______", row[f"Filler{answer}"].lower()))
    return ids, titles, processed_sentence_instances, answer_instances, original_answer_char_positions, prev_instances, now_instances, next_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make multitask dataset")
    parser.add_argument('--path_to_train', type=str, default='../../../data/train/traindata.tsv')
    parser.add_argument('--path_to_train_labels', type=str, default='../../../data/train/trainlabels.tsv')

    parser.add_argument("--path_to_conceptnet_csv_en", type=str, default='../../../conceptnet/assertions-570-en.csv')
    parser.add_argument('--path_to_concept_word', type=str, default='../../../conceptnet/concept_word.txt')
    parser.add_argument('--path_to_lemma_json', type=str, default='../../../conceptnet/lemma_matching.json')

    parser.add_argument('--path_to_test', type=str, default='../../../data/test/testdata.tsv')

    args = parser.parse_args()
    concept_vocab = read_concept_vocab(args)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    matcher = load_matcher(args, nlp)

    # load dataset
    # train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    # training_label_set = pd.read_csv(args.path_to_train_labels, sep="\t", header=None, names=["Id", "Label"])
    conceptnet_dataset = pd.read_csv(args.path_to_conceptnet_csv_en, sep='\t', header=None,
                                     names=["relation", "header", "tail", "weight"])

    test_set = pd.read_csv(args.path_to_test, sep='\t', quoting=3)

    ids, titles, sentences, answers, origianl_answer_char_positions, prev_instances, now_instances, next_instances = retrieve_all_processed_data_from_dataset(test_set)
    # plausible_labels = retrieve_labels_from_dataset_for_classification(training_label_set)

    dataset = []
    for title, sentence, answer in tqdm(zip(titles, sentences, answers), total=len(titles)):
        content = {}
        doc = nlp(sentence)
        matches = matcher(doc)
        sentence_span_concept = []
        for match_id, start, end in matches:
            span = doc[start:end].text
            if span is not None:
                if span in sentence_span_concept:
                    continue
                else:
                    sentence_span_concept.append(span.lower())

        doc = nlp(answer)
        matches = matcher(doc)
        answer_span_concept = []
        for match_id, start, end in matches:
            span = doc[start:end].text
            if span is not None:
                if span in answer_span_concept:
                    continue
                else:
                    answer_span_concept.append(span.lower())
        if len(answer_span_concept) == 0:
            print(answer)
            answer_span_concept.append(answer.lower())
        if len(sentence_span_concept) == 0:
            print(sentence)
