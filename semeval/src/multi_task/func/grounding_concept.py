from tqdm import tqdm
import numpy as np
import argparse, spacy, nltk, json
from spacy.matcher import Matcher
from itertools import permutations, product


#nltk.download('stopwords')

def read_concept_vocab(args):
    with open(args.path_to_concept_word, "r", encoding='utf8') as f:
        concept_vocab = [line.strip() for line in list(f.readlines())]
    concept_vocab = [c.replace("_", " ") for c in concept_vocab]
    return concept_vocab


def create_pattern(doc, stopwords):
    pronoun_list = set(["my", "you", "it", "its", "your","i","he", "she","his","her","they","them","their","our","we"])

    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or all([(token.text in stopwords or token.lemma_ in stopwords) for token in doc]):  #
        return None  # ignore this concept as pattern
    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    return pattern


def make_lemma_json(args, nlp, concept_vocab_list):
    stopwords = nltk.corpus.stopwords.words('english')

    #nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    docs = nlp.pipe(concept_vocab_list)

    all_patterns = {}
    for doc in tqdm(docs, total=len(concept_vocab_list)):
        pattern = create_pattern(doc, stopwords)
        if pattern is None:
            continue
        all_patterns["_".join(doc.text.split(" "))] = pattern
    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(args.path_to_lemma_json, 'w', encoding='utf8') as f:
        json.dump(all_patterns, f)


def load_matcher(args, nlp):
    with open(args.path_to_lemma_json, "r", encoding='utf8') as f:
        all_patterns = json.load(f)
    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
        matcher.add(concept, [pattern], on_match=None)
    return matcher


def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_", " "))
    lemm = set()
    lemm.add("_".join([token.lemma_ for token in doc]))
    return lemm


def ground_mention_concepts(args, nlp, matcher, sentence, answer=""):
    s = sentence.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mention_concepts = set()
    span_to_concepts = {}
    for match_id, start, end in matches:
        span = doc[start:end].text
        if len(set(span.split(" ")).intersection(set(answer.split(" "))))> 0:
            continue
        original_concept = nlp.vocab.strings[match_id]
        if len(original_concept.split("_")) == 1:
            original_concept = list(lemmatize(nlp, original_concept))[0]
        if span not in span_to_concepts:
            span_to_concepts[span] = set()
        span_to_concepts[span].add(original_concept)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)

        shortest = concepts_sorted[0:3]
        for c in shortest:
            lemma = lemmatize(nlp, c)
            intersect = lemma.intersection(shortest)
            if len(intersect)>0:
                mention_concepts.add(list(intersect)[0])
            else:
                mention_concepts.add(c)
    return mention_concepts


def hard_ground(nlp, concept_vocab, sentence):
    sentence = sentence.lower()
    doc = nlp(sentence)
    result = set()
    for chunk in doc:
        if chunk.lemma_ in concept_vocab:
            result.add(chunk.lemma_)
    sentence = "_".join([chunk.text for chunk in doc])
    if sentence in concept_vocab:
        result.add(sentence)
    return result


def match_mention_concepts(args, concept_vocab, nlp, sentences, answers):
    matcher = load_matcher(args, nlp)
    result = []
    print("begin matching concepts")
    for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
        answer = answers[idx]
        all_concepts = ground_mention_concepts(args, nlp, matcher, sentence, answer) #all concepts{'say', 'sometimes_people', 'sometimes', 'stupid', 'people', 'someone'}
        answer_concepts = ground_mention_concepts(args, nlp, matcher, answer) # answer concepts {'swimming_pool', 'swim', 'pool'}
        sentence_concepts = all_concepts - answer_concepts #sentence concepts {'say', 'sometimes_people', 'sometimes', 'stupid', 'people', 'someone'}
        if len(sentence_concepts) == 0:
            sentence_concepts = hard_ground(nlp, concept_vocab, sentence)
        if len(answer_concepts) == 0:
            answer_concepts = hard_ground(nlp, concept_vocab, answer)
        result.append({
            "sentence": sentence,
            "answer": answer,
            "sentence_concepts":list(sentence_concepts),
            "answer_concepts":list(answer_concepts)
        })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use conceptnet 5.7 csv")
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
    args = parser.parse_args()
    concept_vocab = read_concept_vocab(args)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    #make_lemma_json(args, nlp, concept_vocab) ## lemma json 파일 만들때만 사용. matching 시키기 위해서 만듬

    matcher = load_matcher(args, nlp)

    #result = match_mention_concepts(args, concept_vocab, nlp, sentences=["Sometimes people say that someone stupid has no swimming pool."], answers=['swimming pool'])
    #print(result)

    hard_result = hard_ground(nlp, concept_vocab, 'If you put in too much gel, it makes your hair look crunchy and wet.')
    print(hard_result)

    result = ground_mention_concepts(args, nlp, matcher, 'If you put in too much gel, it makes your hair look crunchy and wet.')
    print(result)
    print(len(result))
    sentence_answer = ground_mention_concepts(args, nlp, matcher, 'gel')
    print(sentence_answer)
    pairs = [result, sentence_answer]
    pairs = list(product(*pairs))
    print(pairs)



