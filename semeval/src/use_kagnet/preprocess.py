import argparse
from multiprocessing import cpu_count
from src.use_kagnet.embedding import *
from src.use_kagnet.construct_graph import *
from src.use_kagnet.grounding import *
from src.use_kagnet.paths import *
from src.use_kagnet.subgraph import *
from src.use_kagnet.triples import *

input_paths = {
    'train': {
        'traindata': '../../data/train/traindata.tsv',
        'trainlabels': '../../data/train/trainlabels.tsv',
        'trainsccores': '../../data/train/trainscores.tsv',
        'trainmultitask': '../../data/multitask_dataset_train.json'
    },
    'dev': {
        'devdata': '../../data/devdata.tsv',
        'devlabels': '../../data/devlabels.tsv',
        'devscores': '../../data/devscores.tsv',
        'devmultitask': '../../data/multitask_dataset_dev.json'
    },
    'test': {
        'testdata': '../../data/test/testdata.tsv',
        'testmultitask': '../../data/multitask_dataset_test.json'
    },
    'cpnet': {
        'csv': '../../conceptnet/assertions-570.csv'
    },
    'glove': {
        'txt': '../../glove/glove.6B.300d.txt'
    },
    'numberbatch': {
        'txt': '../../transe/numberbatch-en-19.08.txt'
    },
    'transe': {
        'ent': '../../transe/glove.transe.sgd.ent.npy',
        'rel': '../../transe/glove.transe.sgd.rel.npy'
    }
}

output_path = {
    'cpnet': {
        'csv': '../../other/conceptnet.en.csv',
        'vocab': '../../other/concept.txt',
        'patterns': '../../other/matcher_patterns.json',
        'pruned_graph': '../../other/conceptnet.en.pruned.graph'
    },
    'glove': {
        'npy': '../../other/glove.6B.300d.npy',
        'vocab': '../../other/glove.vocab'
    },
    'numberbatch': {
        'npy': '../../other/nb.npy',
        'vocab': '../../other/nb.vocab',
        'concept_npy': '../../other/concept.nb.npy'
    },
    # task 데이터
    'semeval7': {
        'paths': {
            'raw_train': '../../other/train.paths.raw.jsonl',
            'raw_dev': '../../other/dev.paths.raw.jsonl',
            'raw_test': '../../other/test.paths.raw.jsonl',
            'scores_train': '../../other/train.paths.scores.jsonl',
            'scores_dev': '../../other/dev.paths.scores.jsonl',
            'scores_test': '../../other/test.paths.scores.jsnol',
            'pruned_train': '../../other/train.paths.pruned.jsonl',
            'pruned_dev': '../../other/dev.paths.pruned.jsonl',
            'pruned_test': '../../other/test.paths.pruned.jsonl',
            'adj_train': '../../other/train.paths.adj.jsonl',
            'adj_dev': '../../other/dev.paths.adj.jsonl',
            'adj_test': '../../other/test.paths.adj.jsonl'
        },
        'graph': {
            'train': '../../other/train.graph.jsonl',
            'dev': '../../other/dev.graph.jsonl',
            'test': '../../other/test.graph.jsonl',
            'adj_train': '../../other/train.graph.adj.pk',
            'adj_dev': '../../other/dev.graph.adj.pk',
            'adj_test': '../../other/test.graph.adj.pk',
            'nxg_from_adj_train': '../../other/train.graph.adj.jsonl',
            'nxg_from_adj_dev': '../../other/dev.graph.adj.jsonl',
            'nxg_from_adj_test': '../../other/test.graph.adj.jsonl'
        },
        'triple': {
            'train': '../../other/train.triples.pk',
            'dev': '../../other/dev.triples.pk',
            'test': '../../other/test.triples.pk',
        }
    }
    # statement
    # statement-with-ans-pos ?
    # tokenized
    # grounded
    # paths
    # grpah
    # triple
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['semeval7'], choices=['common', 'semeval7', 'exp', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    print(cpu_count())
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    # find path할때 sentence concept 이런의미로 sc 라고 했는데 그게 train.paths.raw.jsonl 이거에 남았음. 나중에 뭔가 안되면 그거 체크해봐야할듯
    routines = {
        # 'common': [
        #     {'func': glove2npy, 'args': (input_paths['glove']['txt'], output_path['glove']['npy'], output_path['glove']['vocab'])},
        #     {'func': glove2npy, 'args': (input_paths['numberbatch']['txt'], output_path['numberbatch']['npy'], output_path['numberbatch']['vocab'], True)},
        #     {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_path['cpnet']['csv'], output_path['cpnet']['vocab'])},
        #     {'func': load_pretrained_embeddings, 'args': (output_path['numberbatch']['npy'], output_path['numberbatch']['vocab'], output_path['cpnet']['vocab'],
        #      False, output_path['numberbatch']['concept_npy'])},
        #     # generate_graph = 원래 construct_graph
        #     {'func': generate_graph, 'args': (output_path['cpnet']['csv'], output_path['cpnet']['vocab'], output_path['cpnet']['pruned_graph'], True)},
        #     {'func': create_matcher_patterns, 'args': (output_path['cpnet']['vocab'], output_path['cpnet']['patterns'])},
        # ],
        'semeval7': [
            #이미 각 데이터에서 컨셉들 뽑아둔 멀티태스크 데이터 있으니까. kagnet ground까진 있음
            # {'func': find_paths, 'args': (
            #     input_paths['train']['trainmultitask'], output_path['cpnet']['vocab'],
            #     output_path['cpnet']['pruned_graph'], output_path['semeval7']['paths']['raw_train'], args.nprocs, args.seed
            # )},
            # {'func': find_paths, 'args': (
            #     input_paths['dev']['devmultitask'], output_path['cpnet']['vocab'],
            #     output_path['cpnet']['pruned_graph'], output_path['semeval7']['paths']['raw_dev'], args.nprocs,
            #     args.seed
            # )},
            # {'func': find_paths, 'args': (
            #     input_paths['test']['testmultitask'], output_path['cpnet']['vocab'],
            #     output_path['cpnet']['pruned_graph'], output_path['semeval7']['paths']['raw_test'], args.nprocs,
            #     args.seed
            # )},

            # {'func': score_paths, 'args': (
            #     output_path['semeval7']['paths']['raw_train'], input_paths['transe']['ent'], input_paths['transe']['rel'],
            #     output_path['cpnet']['vocab'], output_path['semeval7']['paths']['scores_train'], args.nprocs
            # )},
            # {'func': score_paths, 'args': (
            #     output_path['semeval7']['paths']['raw_dev'], input_paths['transe']['ent'],
            #     input_paths['transe']['rel'],
            #     output_path['cpnet']['vocab'], output_path['semeval7']['paths']['scores_dev'], args.nprocs
            # )},
            # {'func': score_paths, 'args': (
            #     output_path['semeval7']['paths']['raw_test'], input_paths['transe']['ent'],
            #     input_paths['transe']['rel'],
            #     output_path['cpnet']['vocab'], output_path['semeval7']['paths']['scores_test'], args.nprocs
            # )},

            # {'func': prune_paths,
            #  'args': (output_path['semeval7']['paths']['raw_train'], output_path['semeval7']['paths']['scores_train'],
            #           output_path['semeval7']['paths']['pruned_train'], args.path_prune_threshold)},
            # {'func': prune_paths,
            #  'args': (output_path['semeval7']['paths']['raw_dev'], output_path['semeval7']['paths']['scores_dev'],
            #           output_path['semeval7']['paths']['pruned_dev'], args.path_prune_threshold)},
            # {'func': prune_paths,
            #  'args': (output_path['semeval7']['paths']['raw_test'], output_path['semeval7']['paths']['scores_test'],
            #           output_path['semeval7']['paths']['pruned_test'], args.path_prune_threshold)},

            # {'func': generate_subgraph,
            #  'args': (input_paths['train']['trainmultitask'], output_path['semeval7']['paths']['pruned_train'],
            #           output_path['cpnet']['vocab'], output_path['cpnet']['pruned_graph'],
            #           output_path['semeval7']['graph']['train'])},
            # {'func': generate_subgraph,
            #  'args': (input_paths['dev']['devmultitask'], output_path['semeval7']['paths']['pruned_dev'],
            #           output_path['cpnet']['vocab'], output_path['cpnet']['pruned_graph'],
            #           output_path['semeval7']['graph']['dev'])},
            # {'func': generate_subgraph,
            #  'args': (input_paths['test']['testmultitask'], output_path['semeval7']['paths']['pruned_test'],
            #           output_path['cpnet']['vocab'], output_path['cpnet']['pruned_graph'],
            #           output_path['semeval7']['graph']['test'])},
            # {'func': generate_adj_data_from_grounded_concepts,
            #  'args': (input_paths['train']['trainmultitask'], output_path['cpnet']['pruned_graph'],
            #           output_path['cpnet']['vocab'], output_path['semeval7']['graph']['adj_train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts,
            #  'args': (input_paths['dev']['devmultitask'], output_path['cpnet']['pruned_graph'],
            #           output_path['cpnet']['vocab'], output_path['semeval7']['graph']['adj_dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts,
            #  'args': (input_paths['test']['testmultitask'], output_path['cpnet']['pruned_graph'],
            #           output_path['cpnet']['vocab'], output_path['semeval7']['graph']['adj_test'], args.nprocs)},
            # {'func': generate_triples_from_adj,
            #  'args': (output_path['semeval7']['graph']['adj_train'],input_paths['train']['trainmultitask'],
            #           output_path['cpnet']['vocab'], output_path['semeval7']['triple']['train'])},
            # {'func': generate_triples_from_adj,
            #  'args': (output_path['semeval7']['graph']['adj_dev'], input_paths['dev']['devmultitask'],
            #           output_path['cpnet']['vocab'], output_path['semeval7']['triple']['dev'])},
            # {'func': generate_triples_from_adj,
            #  'args': (output_path['semeval7']['graph']['adj_test'], input_paths['test']['testmultitask'],
            #           output_path['cpnet']['vocab'], output_path['semeval7']['triple']['test'])},
            # {'func': generate_path_and_graph_from_adj, 'args': (
            # output_path['semeval7']['graph']['adj_train'], output_path['cpnet']['pruned_graph'],
            # output_path['semeval7']['paths']['adj_train'], output_path['semeval7']['graph']['nxg_from_adj_train'],
            # args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (
            # output_path['semeval7']['graph']['adj_dev'], output_path['cpnet']['pruned_graph'],
            # output_path['semeval7']['paths']['adj_dev'], output_path['semeval7']['graph']['nxg_from_adj_dev'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (
            # output_path['semeval7']['graph']['adj_test'], output_path['cpnet']['pruned_graph'],
            # output_path['semeval7']['paths']['adj_test'], output_path['semeval7']['graph']['nxg_from_adj_test'],
            # args.nprocs)},
        ]
    }

    for routine in args.run:
        for routine_dict in routines[routine]:
            routine_dict['func'](*routine_dict['args'])

if __name__ == '__main__':
    main()

