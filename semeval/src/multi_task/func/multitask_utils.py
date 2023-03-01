import collections
import json
import torch
from torch.utils.data import TensorDataset
from transformers import RobertaTokenizerFast
from transformers import ElectraTokenizer
import pandas as pd
import argparse, re
from tqdm import tqdm
import logging
from spacy.matcher import Matcher
import spacy

from src.multi_task.func.grounding_concept import load_matcher
from src.multi_task.func.my_utils import retrieve_all_instances_from_dataset
from src.multi_task.func.my_utils import retrieve_labels_from_dataset_for_classification
from src.multi_task.func.grounding_concept import read_concept_vocab
logging.basicConfig(level=logging.DEBUG)

def standard_sentence(sentence):
    sentence = sentence.replace("don't", "do not")
    sentence = sentence.replace("—", " ")
    sentence = sentence.replace('\"', '')
    sentence = sentence.replace("(...)", "")
    sentence = sentence.replace("*", "")
    sentence = sentence.replace(";", "")
    sentence= sentence.lower()
    return sentence.strip()


def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_", " "))
    lemm = set()
    lemm.add("_".join([token.lemma_ for token in doc]))
    return lemm


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i,c) in enumerate(text):
        if c == " ":
            continue
        ns_to_s_map[len(ns_chars)] = i
        ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)




if __name__ == "__main__":
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
    parser.add_argument(
        '--max_context_length',
        type=int,
        default=256
    )
    args = parser.parse_args()
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    matcher = load_matcher(args, nlp)
    concept_vocab = read_concept_vocab(args)

    conceptnet_dataset = pd.read_csv(args.path_to_conceptnet, sep='\t', header=None, names=['relation', 'header', 'tail', 'weight'])
    conceptnet_relation = sorted(conceptnet_dataset['relation'].unique().tolist())
    conceptnet_relation.insert(0, 'no relation')
    relation2idx = {}
    idx2relation = {}
    for idx, relation in enumerate(conceptnet_relation):
        relation2idx[relation] = idx
        idx2relation[idx] = relation
    print(relation2idx, idx2relation)

    #print(conceptnet_dataset['relation'].value_counts())
    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    _, article_title, prev_sentences, now_sentences, next_sentences, filler_option = retrieve_all_instances_from_dataset(train_set) #len : 19975개 각각
    training_label_set = pd.read_csv(args.path_to_training_labels, sep="\t", header=None, names=["Id", "Label"])
    plausible_label = retrieve_labels_from_dataset_for_classification(training_label_set)


    with open(args.path_to_multitask_dataset_json, 'r', encoding='utf8') as f:
        multi_dataset = json.load(f)

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

    '''
    context 
    answers
    context_concepts
    answers_concepts
    concept_pairs
    relation_exist
    relation_type
    '''
    sentences_data = [] #19995개로
    answers_data = []
    relation_exist_data = []  #19995개로
    relation_type_data = []  #19995

    sentences_concepts_data = []
    answers_concepts_data = []
    concept_pairs_data = []

    # 데이터 불러오기
    for data in multi_dataset:
        context = data['context']
        context_concepts = data['context_concepts']
        answers_concepts = data['answers_concepts']
        concept_pairs = data['concept_pairs']
        relation_exist = data['relation_exist']
        relation_type = data['relation_type']
        answers = data['answers']
        # for문 안쪽엔 보기로 반복되는 부분 넣고
        for answer in answers:
            # 빈칸에 정답 넣은 문장
            sentence = context.replace("______", answer)
            sentences_data.append(sentence)
            # 정답 추가
            answers_data.append(answer)
            # 관계 있나 없나 추가
            relation_exist_data.append(relation_exist[answer])
            process_relation_type = []
            for relation in relation_type[answer]:
                if relation !=0:
                    process_relation_type.append(relation2idx[relation])
                else:
                    process_relation_type.append(0)
            # 관계 타입 추가
            relation_type_data.append(process_relation_type)
            sentences_concepts_data.append(context_concepts)
            answers_concepts_data.append(answers_concepts[answer])
            concept_pairs_data.append(concept_pairs[answer])

    # 토크나이즈 한거랑 뽑은 컨셉이랑 어떻게 매핑할거냐
    for (sentence, answer) in zip(sentences_data[5:10], answers_data[5:10]):
        sentence = standard_sentence(sentence) # 문장 정규화
        doc = nlp(sentence) #라이브러리 돌리고
        matches = matcher(doc) # lemma랑 시작 끝 인덱스 구하기


        tokenize_sentence = " ".join(tokenizer.tokenize(sentence))
        length_tokenizer_sentence = len(tokenize_sentence)
        doc_tokens = []
        #캐릭터 단위로 공백을 기준으로 자른 문장에서 몇 번째 단어에 있는가
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
            char_to_word_offset.append(len(doc_tokens)-1)

        #mention_concepts = set()
        #span_to_concepts = {}
        #span_start_end = {}
        for match_id, start, end in matches:
            span = doc[start:end].text

            #start_position = sentence_consist_of_tokens.find(span) #char단위로해서 시작 지점
            #end_position = start_position + len(span) - 1 #char단위로해서 끝 지점
            start_position = char_to_word_offset[sentence.find(span)]
            end_position = char_to_word_offset[min(sentence.find(span)+len(span)-1, len(char_to_word_offset)-1)]
            # mrc에서 get final text하는 부분임.
            # (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(sentence)
            # (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)
            #
            # tok_s_to_ns_map = {}
            # for (i, tok_index) in tok_ns_to_s_map.items():
            #     tok_s_to_ns_map[tok_index] = i
            #
            # orig_start_position = None
            # if start_position in tok_s_to_ns_map:
            #     ns_start_position = tok_s_to_ns_map[start_position]
            #     if ns_start_position in orig_ns_to_s_map:
            #         orig_start_position = orig_ns_to_s_map[ns_start_position]
            # orig_end_position = None
            # if end_position in tok_s_to_ns_map:
            #     ns_end_position = tok_s_to_ns_map[end_position]
            #     if ns_end_position in orig_ns_to_s_map:
            #         orig_end_position = orig_ns_to_s_map[ns_end_position]
            # output_text = sentence[orig_start_position: (orig_end_position + 1)]

            tok_to_orig_index = [] # 몇번째 토큰이 몇번쨰 단어에 속하나
            orig_to_tok_index = [] # 몇번쨰 어절이 몇 번째 토큰?
            all_doc_tokens = []

            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = orig_to_tok_index[start_position] #원래는 example.start_posiition
            if end_position < len(doc_tokens)-1:
                tok_end_position = orig_to_tok_index[end_position + 1 ] - 1
            else:
                tok_end_position = len(all_doc_tokens)-1
            temp = all_doc_tokens[tok_start_position:tok_end_position+1]
            # tok_answer_text = " ".join(tokenizer.tokenize(span[start_position:end_position+1]))
            # #mrc: _improve_answer_span. 원래는 orig_answer_text를 넣는다. 그래서 정답 넣어.
            # # 구 형태일때 하는거같은데..
            # for new_start in range(tok_start_position, tok_end_position+1):
            #     for new_end in range(tok_end_position, new_start -1, -1):
            #         text_span = " ".join(all_doc_tokens[new_start:(new_end+1)])
            #         if text_span == tok_answer_text:
            #             test = (new_start, new_end)
            #             temp = all_doc_tokens[new_start:new_end]
            #             print(temp)
            #         else:
            #             test = tok_start_position, tok_end_position
            #             temp = all_doc_tokens[tok_start_position:tok_end_position]
            #             print(temp)


            # truncated_context = tokenizer.encode(sentence, add_special_tokens=False, max_length=args.max_context_length)
            # sequence_added_tokens = (
            #     tokenizer.model_max_length - tokenizer.max_len_single_sentence +1
            #     if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
            #     else tokenizer.model_max_length - tokenizer.max_len_single_sentence
            # )
            # sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
            # span_doc_tokens = all_doc_tokens

            token_to_orig_map = {}
            for i in range(len(all_doc_tokens)):
                index = i
                token_to_orig_map[index] = tok_to_orig_index[i]


            # if len(set(span.split(" ")).intersection(set(answer.split(" ")))) > 0:
            #     continue
            # original_concept = nlp.vocab.strings[match_id]
            # if len(original_concept.split("_")) == 1:
            #     original_concept = list(lemmatize(nlp, original_concept))[0]
            # if span not in span_to_concepts:
            #     span_to_concepts[span] = set()
            #     span_start_end[span] = set()
            # span_to_concepts[span].add(original_concept)
            #
            # for mat in re.finditer(span, sentence):
            #     span_start_end[span].add((mat.start(), mat.end()))

        # for span, concepts in span_to_concepts.items():
        #     concepts_sorted = list(concepts)
        #     concepts_sorted.sort(key=len)
        #
        #     shortest = concepts_sorted[0:3]
        #     for c in shortest:
        #         lemma = lemmatize(nlp, c)
        #         intersect = lemma.intersection(shortest)
        #         if len(intersect) > 0:
        #             mention_concepts.add(list(intersect)[0])
        #         else:
        #             mention_concepts.add(c)

    # print(
    #     len(sentences_data),
    #     len(answers_data),
    #     len(relation_exist_data),
    #     len(relation_type_data),
    #     len(sentences_concepts_data),
    #     len(answers_concepts_data),
    #     len(concept_pairs_data)
    # )
    # test_pair = concept_pairs_data[:10]
    # for sentence in sentences_data[:10]:
    #     #sentence = standard_sentence(sentence)
    #     #sentence = re.sub(r'\d+.\w', r'\d+. \w', sentence)
    #     doc = nlp(sentence)
    #     matches = matcher(doc)  #[(13950701771273664180, 1, 2), (7886921362353864828, 1, 2), (8262243250093128877, 1, 2), (18211330135299466081, 1, 2), (8565691695081562185, 1, 3), (6363481171083395766, 1, 3), (1562404752343807973, 1, 3), (4121829478905990290, 2, 3), (9515483755171332513, 2, 3), (1287765359399436221, 2, 3), (9749302340701671122, 2, 3), (16769619252953371261, 2, 3), (5379644128261274187, 4, 5), (6878210874361030284, 5, 6), (7119356634073270832, 5, 6), (6873750497785110593, 8, 9), (16421957100465448365, 8, 9), (17544535479879614370, 8, 9), (15169213402950300534, 8, 9), (9652663489128248624, 9, 10), (13336162231634412336, 9, 10), (15585516444165348005, 9, 10), (10420837567980902426, 9, 10), (17461235395181654430, 11, 12), (14402036221983944882, 11, 12), (13367314321634499193, 11, 12), (245637087864966835, 11, 12), (13950701771273664180, 13, 14), (7886921362353864828, 13, 14), (8262243250093128877, 13, 14), (18211330135299466081, 13, 14), (8565691695081562185, 13, 15), (6363481171083395766, 13, 15), (1562404752343807973, 13, 15), (4121829478905990290, 14, 15), (9515483755171332513, 14, 15), (1287765359399436221, 14, 15), (9749302340701671122, 14, 15), (16769619252953371261, 14, 15), (13950701771273664180, 17, 18), (7886921362353864828, 17, 18), (8262243250093128877, 17, 18), (18211330135299466081, 17, 18), (8565691695081562185, 17, 19), (6363481171083395766, 17, 19), (1562404752343807973, 17, 19), (4121829478905990290, 18, 19), (9515483755171332513, 18, 19), (1287765359399436221, 18, 19), (9749302340701671122, 18, 19), (16769619252953371261, 18, 19), (4582038485178338594, 17, 20), (1826119438242743099, 19, 20), (12462445139066697090, 19, 20), (14937911803315884794, 19, 20), (2089162579411277415, 19, 20), (15991714103461148278, 19, 20), (16423567875008657439, 19, 20), (8863402319515883670, 21, 22), (2754663265045775480, 21, 22), (12291142353903974928, 21, 22), (8188770785677902486, 21, 22)]
    #     print(sentence)
    #     process_sentence = tokenizer(sentence)
    #     print(process_sentence)
    #
    #     input_ids = process_sentence['input_ids']
    #     tokens_sentence = process_sentence.encodings[0].tokens
    #     for chunk in doc:
    #         if chunk.lemma_ in concept_vocab:
    #             print(chunk)
    #
    #     for match_id, start, end in matches:
    #         span = doc[start:end].text
    #         print('span:', span)
    #         original_concept = nlp.vocab.strings[match_id]
    #         #rint(original_concept)
    #         print('puls 1:', tokenizer.decode(input_ids[start+1:end+1]))
