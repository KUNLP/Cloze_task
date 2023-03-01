from multiprocessing import Pool
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
import json

nltk_stopwords = nltk.corpus.stopwords.words('english')


def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab


def create_pattern(nlp, doc, debug=False):
    pronoun_list = set(["my", "you", "it", "its", "your", "i", "he", "she", "his", "her", "they", "them", "their", "our", "we"])
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or \
            all([(token.text in nltk_stopwords or token.lemma_ in nltk_stopwords) for token
                 in doc]):
        if debug:
            return False, doc.text
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    if debug:
        return True, doc.text
    return pattern


def create_matcher_patterns(cpnet_vocab_path, output_path, debug=False):
    cpnet_vocab = load_cpnet_vocab(cpnet_vocab_path)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    docs = nlp.pipe(cpnet_vocab)
    all_patterns = {}

    if debug:
        f = open("filtered_concept.txt", "w")
    for doc in tqdm(docs, total=len(cpnet_vocab)):

        pattern = create_pattern(nlp, doc, debug)
        if debug:
            if not pattern[0]:
                f.write(pattern[1] + '\n')

        if pattern is None:
            continue
        all_patterns["_".join(doc.text.split(" "))] = pattern

    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    if debug:
        f.close()