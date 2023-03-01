import argparse, logging
import os

from src.multi_task.func.multi_task_main_function import dev, test, test_test
from src.multi_task.func.use_paper_func import train
import wandb

# def main(args):
#     wandb.init()
#
#     print(args.mode)
#     if args.mode == 'train':
#         train(args)
#     elif args.mode == 'dev':
#        dev(args)
#     elif args.mode == 'test':
#         test(args)
#     else:
#         test_test(args)

def main(args):
    if args.mode == 'train':
        train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--path_to_train', type=str, default='../../data/train/traindata.tsv')
    parser.add_argument('--path_to_train_labels', type=str, default='../../data/train/trainlabels.tsv')
    parser.add_argument('--path_to_dev', type=str, default='../../data/dev/devdata.tsv')
    parser.add_argument('--path_to_dev_labels', type=str, default='../../data/dev/devlabels.tsv')
    parser.add_argument('--path_to_test', type=str, default='../../data/test/testdata.tsv')

    parser.add_argument('--path_to_multitask_dataset_train', type=str, default='../../data/multitask_dataset_train.json')
    parser.add_argument('--path_to_multitask_dataset_dev', type=str, default='../../data/multitask_dataset_dev.json')
    parser.add_argument('--path_to_multitask_dataset_test', type=str, default='../../data/multitask_dataset_test.json')
    parser.add_argument('--path_to_conceptnet', type=str, default='../../conceptnet/assertions-570-en.csv')
    parser.add_argument('--path_to_lemma_json', type=str, default='../../conceptnet/lemma_matching.json')
    parser.add_argument('--path_to_concept_word', type=str, default='../../conceptnet/concept_word.txt')

    parser.add_argument('--plausible_num_label', type=int, default=3)
    parser.add_argument('--relation_exist_label', type=int, default=2)
    parser.add_argument('--relation_type_label', type=int, default=18)

    parser.add_argument('--input_sentence_max_length', type=int, default=256)
    parser.add_argument('--max_length_pair_token', type=int, default=177)
    parser.add_argument('--relation_exist_padding_idx', type=int, default=2)
    parser.add_argument('--relation_type_padding_idx', type=int, default=18)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    parser.add_argument("--relation2idx", type=dict, default={'no relation': 0, 'antonym': 1, 'atlocation': 2, 'capableof': 3, 'causes': 4, 'createdby': 5, 'desires': 6, 'hascontext': 7, 'hasproperty': 8, 'hassubevent': 9, 'isa': 10, 'madeof': 11, 'notcapableof': 12, 'notdesires': 13, 'partof': 14, 'receivesaction': 15, 'relatedto': 16, 'usedfor': 17})
    parser.add_argument('--idx2relation', type=dict, default={0: 'no relation', 1: 'antonym', 2: 'atlocation', 3: 'capableof', 4: 'causes', 5: 'createdby', 6: 'desires', 7: 'hascontext', 8: 'hasproperty', 9: 'hassubevent', 10: 'isa', 11: 'madeof', 12: 'notcapableof', 13: 'notdesires', 14: 'partof', 15: 'receivesaction', 16: 'relatedto', 17: 'usedfor'})

    parser.add_argument("--output_dir", type=str, default="../../output")
    parser.add_argument("--load_my_sota", type=str, default="../../output/electra-large_checkpoint_1/")
    parser.add_argument("--save_pickle_train_data", type=str, default='../../data/save_pickle_train_data.pkl')
    parser.add_argument("--save_pickle_dev_data", type=str, default='../../data/save_pickle_dev_data.pkl')
    parser.add_argument("--save_pickle_test_data", type=str, default='../../data/save_pickle_test_data.pkl')
    parser.add_argument("--PLM", type=str, default='electra-large')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'])
    parser.add_argument("--subtask", type=str, default='classification')
    parser.add_argument("--path_to_predictions", type=str, default='../../data/predictions/pred_test_class.tsv')

    #parser.add_argument('--rcgn_model_path', type=str, default='/home/wonjae/workspace/krqa/MHGRN/saved_models/csqa.bert-large-uncased.rgcn/')
    args = parser.parse_args()

    main(args)
