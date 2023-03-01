import argparse, re, string
import pandas as pd
from tqdm import tqdm
from itertools import product
from src.multi_task.func.my_utils import retrieve_all_instances_from_dataset, retrieve_labels_from_dataset_for_classification


def normalize_sentence(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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
        '--path_to_knowledge',
        type=str,
        default='../../../data/transomcs/TransOMCS_full.txt'
    )

    args = parser.parse_args()

    #load dataset
    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    training_label_set = pd.read_csv(args.path_to_training_labels, sep="\t", header=None, names=["Id", "Label"])
    knowledge_source = pd.read_csv(args.path_to_knowledge, sep='\t', header=None,
                                   names=['header', 'relation', 'tail', 'score'])  # transomcs KB
    #process data
    _, article_titles, prev_sentences, now_sentences, next_sentences, filler_options = retrieve_all_instances_from_dataset(train_set) #len : 19975개 각각
    plausible_labels = retrieve_labels_from_dataset_for_classification(training_label_set)

    content_sentences = [prev_sentence+now_sentence+next_sentence for (prev_sentence, now_sentence, next_sentence) in zip(prev_sentences, now_sentences, next_sentences)]


    article_titles = [normalize_sentence(article_title) for article_title in article_titles]
    content_sentence = [normalize_sentence(content_sentence) for content_sentence in content_sentences]

