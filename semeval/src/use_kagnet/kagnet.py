import random
import argparse
from transformers.optimization import get_constant_schedule, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.optim import SGD, Adam, AdamW, RAdam
import torch
from src.use_kagnet.modeling_kagnet import *
import time
#from utils.utils import *

OPTIMIZER_CLASSES = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
    'radam': RAdam,
}

ENCODER_DEFAULT_LR = {
    'default': 1e-3,
    'csqa': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'semeval7': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    }
}

EMB_PATHS = {
    'transe': '../../other/glove.transe.sgd.ent.npy',
    'lm': '../../other/glove.transe.sgd.ent.npy',
    'numberbatch': '../../other/concept.nb.npy',
    'tzw': '../../other/tzw.ent.npy',
}

DATASET_SETTING = {
    'csqa': 'inhouse',
    'obqa': 'official',
    'socialiqa': 'official',
    'semeval7': 'official'
}

DATASET_NO_TEST = ['socialiqa']


def get_lstm_config_from_args(args):
    lstm_config = {
        'hidden_size': args.encoder_dim,
        'output_size': args.encoder_dim,
        'num_layers': args.encoder_layer_num,
        'bidirectional': args.encoder_bidir,
        'emb_p': args.encoder_dropoute,
        'input_p': args.encoder_dropouti,
        'hidden_p': args.encoder_dropouth,
        'pretrained_emb_or_path': args.encoder_pretrained_emb,
        'freeze_emb': args.encoder_freeze_emb,
        'pool_function': args.encoder_pooler,
    }
    return lstm_config


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def add_additional_arguments(parser):
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)


def add_optimization_arguments(parser):
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'cross_entropy'], help='model type')
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=100, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=2, type=int, help='stop training if dev does not increase for N epochs')


def add_encoder_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='bert-large-uncased', help='encoder type')
    parser.add_argument('--encoder_layer', default=-1, type=int, help='encoder layer ID to use as features (used only by non-LSTM encoders)')
    parser.add_argument('-elr', '--encoder_lr', default=2e-5, type=float, help='learning rate')
    # used only for LSTM encoder
    parser.add_argument('--encoder_dim', default=128, type=int, help='number of LSTM hidden units')
    parser.add_argument('--encoder_layer_num', default=2, type=int, help='number of LSTM layers')
    parser.add_argument('--encoder_bidir', default=True, type=bool_flag, nargs='?', const=True, help='use BiLSTM')
    parser.add_argument('--encoder_dropoute', default=0.1, type=float, help='word dropout')
    parser.add_argument('--encoder_dropouti', default=0.1, type=float, help='dropout applied to embeddings')
    parser.add_argument('--encoder_dropouth', default=0.1, type=float, help='dropout applied to lstm hidden states')
    parser.add_argument('--encoder_pretrained_emb', default='../../other/glove.6B.300d.npy', help='path to pretrained emb in .npy format')
    parser.add_argument('--encoder_freeze_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze lstm input embedding layer')
    parser.add_argument('--encoder_pooler', default='max', choices=['max', 'mean'], help='pooling function')
    args, _ = parser.parse_known_args()
    parser.set_defaults(encoder_lr=ENCODER_DEFAULT_LR[args.dataset].get(args.encoder, ENCODER_DEFAULT_LR['default']))


def add_data_arguments(parser):
    # arguments that all datasets share
    parser.add_argument('--ent_emb', default=['transe'], choices=['transe', 'numberbatch', 'lm', 'tzw'], nargs='+', help='sources for entity embeddings')
    parser.add_argument('--ent_emb_paths', default=['../../other/glove.transe.sgd.ent.npy'], choices=['./data/cpnet/tzw.ent.npy'], nargs='+', help='paths to entity embedding file(s)')
    parser.add_argument('--rel_emb_path', default='../../other/glove.transe.sgd.rel.npy', help='paths to relation embedding file')
    # dataset specific
    parser.add_argument('-ds', '--dataset', default='semeval7', help='dataset name')
    # statements
    parser.add_argument('--train_statements', default='../../data/multitask_dataset_train.json')
    parser.add_argument('--dev_statements', default='../../data/multitask_dataset_dev.json')
    parser.add_argument('--test_statements', default='../../data/multitask_dataset_test.json')
    # labels
    parser.add_argument('--train_labels', default='../../data/train/trainlabels.tsv')
    parser.add_argument('--dev_labels', default='../../data/dev/devlabels.tsv')
    # preprocessing options
    parser.add_argument('-sl', '--max_seq_len', default=64, type=int)
    parser.add_argument('--format', default=[], choices=['add_qa_prefix', 'no_extra_sep', 'fairseq', 'add_prefix_space'], nargs='*')
    # set dataset defaults
    args, _ = parser.parse_known_args()
    parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb])
    data_splits = ('train', 'dev') if args.dataset in DATASET_NO_TEST else ('train', 'dev', 'test')
    for split in data_splits:
        for attribute in ('statements',):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset)})
    if 'test' not in data_splits:
        parser.set_defaults(test_statements=None)

def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(add_help=False)
    add_data_arguments(parser)
    add_encoder_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)
    return parser

def evaluate_accuracy(eval_set, model, model_type):
    n_correct = 0
    model.eval()
    with torch.no_grad():
        for statements, correct_labels, graphs, cpt_paths, rel_paths, qa_pairs, concept_mapping_dicts, qa_path_num, path_len in eval_set:
            batch_size, num_choice = statements.size(0), statements.size(1)
            flat_statements = statements.view(batch_size * num_choice, -1)
            flat_qa_pairs = sum(qa_pairs, [])
            flat_cpt_paths = sum(cpt_paths, [])
            flat_rel_paths = sum(rel_paths, [])
            flat_qa_path_num = sum(qa_path_num, [])
            flat_path_len = sum(path_len, [])
            if model_type == 'kagnet':
                flat_logits = model(flat_statements, flat_qa_pairs, flat_cpt_paths, flat_rel_paths, graphs,
                                    concept_mapping_dicts, flat_qa_path_num, flat_path_len)
            elif model_type == 'kernet':
                flat_logits = model(flat_statements, flat_qa_pairs, flat_cpt_paths, flat_rel_paths, flat_qa_path_num, flat_path_len)
            elif model_type == 'relation_net':
                flat_logits = model(flat_statements, flat_qa_pairs)
            elif model_type == 'gcn':
                flat_logits = model(flat_statements, graphs)
            elif model_type == 'bert':
                flat_logits = model(flat_statements)
            flat_logits = flat_logits.view(-1, num_choice)
            _, pred = flat_logits.max(1)
            n_correct += (pred == correct_labels).sum().item()
    model.train()
    return n_correct / len(eval_set)


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'../../other/saved_models/{args.dataset}.{args.encoder}.kagnet/', help='model output directory')

    # datasets
    parser.add_argument('--train_graphs', default='../../other/train.graph.adj.jsonl')
    parser.add_argument('--train_paths', default='../../other/train.paths.adj.jsonl')
    parser.add_argument('--dev_graphs', default='../../other/dev.graph.adj.jsonl')
    parser.add_argument('--dev_paths', default='../../other/dev.paths.adj.jsonl')
    parser.add_argument('--test_graphs', default='../../other/test.graph.adj.jsonl')
    parser.add_argument('--test_paths', default='../../other/test.paths.adj.jsonl')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('--lstm_dim', default=128, type=int, help='number of LSTM hidden units')
    parser.add_argument('--lstm_layer_num', default=1, type=int, help='number of LSTM layers')
    parser.add_argument('--bidirect', default=False, type=bool_flag, nargs='?', const=True, help='use bidirectional LSTM')
    parser.add_argument('--qas_encoded_dim', default=128, type=int, help='dimensionality of an encoded (qc, ac) pair')
    parser.add_argument('--num_random_paths', default=None, type=int, help='number random paths to sample during training')
    parser.add_argument('--graph_hidden_dim', default=50, type=int, help='number of hidden units of the GCN')
    parser.add_argument('--graph_output_dim', default=25, type=int, help='number of output units of the GCN')
    parser.add_argument('--freeze_ent_emb', default=False, type=bool_flag, nargs='?', const=True, help='freeze embedding layer')
    parser.add_argument('--freeze_lstm_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze lstm input embedding layer')
    parser.add_argument('--path_attention', default=True, type=bool_flag, nargs='?', const=True, help='use bidirectional LSTM')
    parser.add_argument('--qa_attention', default=True, type=bool_flag, nargs='?', const=True, help='use bidirectional LSTM')

    # regularization
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout probability')

    # other model options
    parser.add_argument('--max_path_len', default=4, type=int)

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=4, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    print()

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    with open(log_path, 'w', encoding='utf-8') as fout:
        fout.write('step,train_acc,dev_acc\n')

    dic = {'transe': 0, 'numberbatch': 1, 'tzw': 2}
    cp_emb, rel_emb = [np.load(args.ent_emb_paths[0])], np.load(args.rel_emb_path)
    #cp_emb, rel_emb = [np.load(args.ent_emb_paths[dic[source]]) for source in args.ent_emb], np.load(args.rel_emb_path)
    cp_emb = np.concatenate(cp_emb, axis=1)
    cp_emb = torch.tensor(cp_emb)
    rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
    rel_emb = torch.tensor(rel_emb)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('num_concepts: {}, concept_dim: {}'.format(concept_num, concept_dim))
    relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
    print('num_relations: {}, relation_dim: {}'.format(relation_num, relation_dim))

    try:

        device0 = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        dataset = KagNetDataLoader(args.train_statements, args.train_labels, args.train_paths, args.train_graphs,
                                   args.dev_statements, args.dev_labels, args.dev_paths, args.dev_graphs,
                                   args.test_statements, args.test_paths, args.test_graphs,
                                   batch_size=args.mini_batch_size, eval_batch_size=args.eval_batch_size, device=device0,
                                   model_name=args.encoder, max_seq_length=args.max_seq_len, max_path_len=args.max_path_len,
                                   use_cache=args.use_cache, format=args.format)
        print('dataset done')

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################
        lstm_config = get_lstm_config_from_args(args)

        model = LMKagNet(model_name=args.encoder, concept_dim=concept_dim, relation_dim=relation_dim, concept_num=concept_num,
                         relation_num=relation_num, qas_encoded_dim=args.qas_encoded_dim, pretrained_concept_emb=cp_emb,
                         pretrained_relation_emb=rel_emb, lstm_dim=args.lstm_dim, lstm_layer_num=args.lstm_layer_num, graph_hidden_dim=args.graph_hidden_dim,
                         graph_output_dim=args.graph_output_dim, dropout=args.dropout, bidirect=args.bidirect, num_random_paths=args.num_random_paths,
                         path_attention=args.path_attention, qa_attention=args.qa_attention, encoder_config=lstm_config)
        print('model done')
        if args.freeze_ent_emb:
            freeze_net(model.decoder.concept_emb)
        print('freezed')
        model.encoder.to(device0)
        print('encoder done')
        model.decoder.to(device0)
        print('decoder done')
    except RuntimeError as e:
        print(e)
        print('best dev acc: 0.0 (at epoch 0)')
        print('final test acc: 0.0')
        print()
        return

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        else:
            print('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    print()
    print('-' * 71)
    global_step, last_best_step = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)
    try:
        for epoch_id in range(args.n_epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    print(00)
                    b = min(a + args.mini_batch_size, bs)
                    logits, _ = model(*[x for x in input_data], layer_id=args.encoder_layer)

                    if args.loss == 'margin_rank':
                        num_choice = logits.size(1)
                        flat_logits = logits.view(-1)
                        correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
                        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
                        wrong_logits = flat_logits[correct_mask == 0]  # of length batch_size*(num_choice-1)
                        y = wrong_logits.new_ones((wrong_logits.size(0),))
                        loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
                    elif args.loss == 'cross_entropy':
                        loss = loss_func(logits, labels[a:b])
                    loss = loss * (b - a) / bs
                    loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                    total_loss = 0
                    start_time = time.time()

                if (global_step + 1) % args.eval_interval == 0:
                    model.eval()
                    dev_acc = evaluate_accuracy(dataset.dev(), model)
                    test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
                    print('-' * 71)
                    print('| step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(global_step, dev_acc, test_acc))
                    print('-' * 71)
                    with open(log_path, 'a') as fout:
                        fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
                    if dev_acc >= best_dev_acc:
                        best_dev_acc = dev_acc
                        final_test_acc = test_acc
                        last_best_step = global_step
                        torch.save([model, args], model_path)
                        print(f'model saved to {model_path}')
                    model.train()
                    start_time = time.time()

                global_step += 1
                # if global_step >= args.max_steps or global_step - last_best_step >= args.max_steps_before_stop:
                #     end_flag = True
                #     break
    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev acc: {:.4f} (at step)'.format(best_dev_acc, last_best_step))
    print('final test acc: {:.4f}'.format(final_test_acc))


def eval(args):
    raise NotImplementedError()  # TODO


if __name__ == '__main__':
    main()
