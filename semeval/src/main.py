"""A module for running baseline models.

Examples:
python main.py --path_to_train train_data.tsv --path_to_training_labels train_labels.tsv --path_to_dev dev_data.tsv --path_to_dev_labels dev_labels.tsv --path_to_predictions pred_dev_class.tsv --classification_baseline bag-of-words
python main.py --path_to_train train_data.tsv --path_to_training_labels train_scores.tsv --path_to_dev dev_data.tsv --path_to_dev_labels dev_scores.tsv --path_to_predictions pred_dev_rank.tsv --ranking_baseline bag-of-words
"""
import argparse
import logging
import pandas as pd
import os
from attrdict import AttrDict
from data import (
    retrieve_instances_from_dataset,
    retrieve_labels_from_dataset_for_classification,
    retrieve_labels_from_dataset_for_ranking,
    write_predictions_to_file,
)
from format_checker_for_dataset import check_format_of_dataset
from format_checker_for_submission import check_format_of_submission
from models import BowClassificationBaseline, BowRankingBaseline
from scorer import score
from utils import retrieve_all_instances_from_dataset
from main_function import train, dev, test

logging.basicConfig(level=logging.DEBUG)


def main(cli_args):
    args = AttrDict(vars(cli_args))
    if args.mode == 'train':
        train(args)
    elif args.mode == 'dev':
        dev(args)
    else:
        test(args)
    # logging.debug(f"Read training dataset from file {args['path_to_train']}")
    # # quoting=3 큰 따음표 무시
    # train_set = pd.read_csv(args["path_to_train"], sep="\t", quoting=3)
    #
    # # 데이터셋이 멀쩡한지 확인하는 정도
    # check_format_of_dataset(train_set)
    #
    # # _ : ['1_1', '1_2', '1_3'] ...
    # # training_instances = ['* Lose weight Fast with stress', '* Lose weight Fast with plenty', '* Lose weight Fast with strengh', '* Lose weight Fast with ease', ...]
    # # 문장에 보기 1~5 채워서 나온것들이 반환된다.
    # _, training_instances = retrieve_instances_from_dataset(train_set)
    #
    # _, prev_sentence, now_sentence, next_sentence = retrieve_all_instances_from_dataset(train_set)
    #
    # logging.debug(f"Read dev dataset from file {args['path_to_dev']}")
    # dev_set = pd.read_csv(args["path_to_dev"], sep="\t", quoting=3)
    # check_format_of_dataset(dev_set)
    # dev_ids, dev_instances = retrieve_instances_from_dataset(dev_set)



    # # Run the baseline
    # if args["classification_baseline"] or args["ranking_baseline"]:
    #     subtask = "classification" if args["classification_baseline"] else "ranking"
    #
    #     logging.debug(
    #         f"Read gold labels for training dataset from file {args['path_to_training_labels']}"
    #     )
    #     training_label_set = pd.read_csv(
    #         args["path_to_training_labels"], sep="\t", header=None, names=["Id", "Label"]
    #     )
    #     check_format_of_submission(training_label_set, subtask=subtask)
    #     baseline_model = None
    #
    #     if (
    #         args["classification_baseline"]
    #         and args["classification_baseline"] == "bag-of-words"
    #     ):
    #         logging.debug("Subtask A: multi-class classification")
    #         logging.debug("Run classification baseline with bag of words")
    #         baseline_model = BowClassificationBaseline()
    #         training_labels = retrieve_labels_from_dataset_for_classification(
    #             training_label_set
    #         )
    #
    #     elif args["ranking_baseline"] and args["ranking_baseline"] == "bag-of-words":
    #         logging.debug("Subtask B: ranking")
    #         logging.debug("Run ranking baseline with bag of words")
    #         baseline_model = BowRankingBaseline()
    #         training_labels = retrieve_labels_from_dataset_for_ranking(training_label_set)
    #
        # dev_predictions = baseline_model.run_held_out_evaluation(
        #     training_instances=training_instances,
        #     training_labels=training_labels,
        #     dev_instances=dev_instances,
        # )
        # prediction_dataframe = write_predictions_to_file(
        #     path_to_predictions=args["path_to_predictions"],
        #     ids=dev_ids,
        #     predictions=dev_predictions,
        #     subtask=subtask,
        # )
    #
    #     logging.debug("Score predictions for dev set")
    #     score(
    #         submission_file=args["path_to_predictions"],
    #         reference_file=args["path_to_dev_labels"],
    #         subtask=subtask,
    #     )


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(__file__))
    print(root_path)
    # Initialize argparse
    ap = argparse.ArgumentParser(description="Run baselines for SemEval-2022 Task 7.")

    # path
    #help="Path to the train instances in tsv format.",
    ap.add_argument("--path_to_train", type=str, default='../data/train/traindata.tsv')
    #help="Path to the labels for the training instances in tsv format.",
    ap.add_argument("--path_to_training_labels", type=str, default='../data/train/trainlabels.tsv')
    #help = "Path to the dev instances in tsv format.",
    ap.add_argument("--path_to_dev", type=str, default='../data/dev/devdata.tsv')
    #help="Path to the labels for the dev instances in tsv format.",
    ap.add_argument("--path_to_dev_labels", type=str, default='../data/dev/devlabels.tsv')
    #help = "Path to tsv file in which to write the predictions",
    ap.add_argument("--path_to_predictions", type=str, default='../data/predictions/pred_dev_class.tsv')
    #help = "Select a baseline classifier: bag-of-words"
    ap.add_argument("--classification_baseline", type=str,default='bag-of-words')
    #help="Select a baseline ranking model: bag-of-words",
    ap.add_argument("--ranking_baseline", type=str, default='bag-of-words')
    #ap.add_argument("--path_to_predictions", type=str, default='../data/predictions/')

    #model
    ap.add_argument("--output_dir", type=str, default='../output/')
    ap.add_argument("--PMmodel", type=str, default='electra')
    ap.add_argument("--checkpoint", type=int, default=0)

    #training
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--num_labels", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # run
    ap.add_argument("--mode", type=str, default='train')
    ap.add_argument("--subtask", type=str, default='classification')


    cli_args = ap.parse_args()
    main(cli_args)