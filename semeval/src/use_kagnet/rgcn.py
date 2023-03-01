import random

from transformers.optimization import get_constant_schedule, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from src.use_kagnet.modeling_rgcn import *
from torch.optim import SGD, Adam, AdamW, RAdam
import argparse
import os
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score

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
        'electra-large': 2e-5
    }
}
EMB_PATHS = {
    'transe': '../../other/glove.transe.sgd.ent.npy',
    'lm': '../../other/glove.transe.sgd.ent.npy',
    'numberbatch': '../../other/concept.nb.npy',
    'tzw': '../../other/tzw.ent.npy',
}
DATASET_NO_TEST = ['socialiqa']


DECODER_DEFAULT_LR = {'csqa': 1e-3, 'obqa': 1e-3, 'semeval7': 2e-5}
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


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    pred_label = []
    corr_label = []
    with torch.no_grad():
        for qids, labels, *input_data in eval_set:
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
            pred_label.extend(logits.argmax(-1).cpu().detach().numpy().tolist())
    return n_correct / n_samples

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
    parser.add_argument('--optim', default='adamw', choices=['sgd', 'adam', 'adamw', 'radam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='warmup_linear', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=100, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=2, type=int, help='stop training if dev does not increase for N epochs')


def add_encoder_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='google/electra-large-discriminator', help='encoder type')
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
    parser.add_argument('-sl', '--max_seq_len', default=256, type=int)
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

def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/{args.dataset}.{args.encoder}.rgcn/', help='model output directory')

    # data
    parser.add_argument('--cpnet_vocab_path', default='../../other/concept.txt')
    parser.add_argument('--num_relation', default=35, type=int, help='number of relations')
    parser.add_argument('--train_adj', default='../../other/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default='../../other/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default='../../other/test.graph.adj.pk')
    parser.add_argument('--train_origin_path', default='../../data/train/traindata.tsv')
    parser.add_argument('--dev_origin_path', default='../../data/dev/devdata.tsv')
    parser.add_argument('--test_origin_path', default='../../data/test/testdata.tsv')

    # model architecture
    parser.add_argument('--ablation', default=[], choices=['no_node_type_emb', 'no_lm'], help='run ablation test')
    parser.add_argument('--diag_decompose', default=False, type=bool_flag, nargs='?', const=True, help='use diagonal decomposition')
    parser.add_argument('--num_basis', default=8, type=int, help='number of basis (0 to disable basis decomposition)')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_layer_num', default=2, type=int, help='number of GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='hidden dim of the fully-connected layers')
    parser.add_argument('--fc_layer_num', default=1, type=int, help='number of the fully-connected layers')
    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--dropoutg', type=float, default=0.1, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.3, help='dropout for fully-connected layers')
    parser.add_argument('--cpt_out_dim', type=int, default=100, help='num of dimension for concepts in processing')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=32, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument('--save', type=bool_flag, default=False, help='whether to save logs and models')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'pred':
        pred(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print('configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    print()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')

    # with open(log_path, 'w', encoding='utf-8') as fout:
    #     fout.write('step,train_acc,dev_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################

    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, axis=1))

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('num_concepts: {}, concept_dim: {}'.format(concept_num, concept_dim))

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        dataset = LMRGCNDataLoader(train_statement_path=args.train_statements, train_label_path=args.train_labels, train_adj_path=args.train_adj, train_origin_path=args.train_origin_path,
                                   dev_statement_path=args.dev_statements, dev_label_path=args.dev_labels, dev_adj_path=args.dev_adj, dev_origin_path=args.dev_origin_path,
                                   test_statement_path=args.test_statements, test_adj_path=args.test_adj, test_origin_path=args.test_origin_path,
                                   batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                                   model_name=args.encoder, max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                   format=args.format)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################

        lstm_config = get_lstm_config_from_args(args)
        model = LMRGCN(args.encoder, num_concepts=concept_num, num_relations=args.num_relation, num_basis=args.num_basis,
                       concept_dim=args.cpt_out_dim, concept_in_dim=concept_dim, num_gnn_layers=args.gnn_layer_num,
                       num_attention_heads=args.att_head_num, fc_dim=args.fc_dim, num_fc_layers=args.fc_layer_num,
                       p_gnn=args.dropoutg, p_fc=args.dropoutf, freeze_ent_emb=args.freeze_ent_emb,
                       pretrained_concept_emb=cp_emb, diag_decompose=args.diag_decompose, ablation=args.ablation, encoder_config=lstm_config)
        if args.freeze_ent_emb:
            freeze_net(model.decoder.concept_emb)
        model.to(device)
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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

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
        loss_func = nn.CrossEntropyLoss()

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print('-' * 71)
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    pred_label = []
    correct_label = []
    start_time = time.time()
    model.train()
    #freeze_net(model.encoder)
    try:
        for epoch_id in range(args.n_epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            model.train()
            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                    pred_label.extend(logits.argmax(dim=-1).cpu().detach().numpy().tolist())
                    correct_label.extend(labels[a:b].view(-1).cpu().detach().numpy().tolist())
                    #total_logit.append(logits.item())
                    if args.loss == 'cross_entropy':
                        # 이 로스 부분이 문제? logits 이 (1,1) 로 나옴.
                        #temp = torch.unsqueeze(labels[a:b], dim=1)
                        # logit size (4,1) / temp size (4,) 이런 형식이니까 형태가 안맞나
                        loss = loss_func(logits, labels[a:b].view(-1))
                    loss = loss * (b - a) / bs
                    loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} | accuracy {}'.format(global_step, scheduler.get_last_lr()[0], total_loss, ms_per_batch, accuracy_score(correct_label, pred_label)))
                    total_loss = 0
                    start_time = time.time()
                global_step += 1

            model.eval()
            dev_acc = evaluate_accuracy(dataset.dev(), model)
            print('-' * 71)
            print('| step {:5} | dev_acc {:7.4f} |'.format(global_step, dev_acc))
            print('-' * 71)
            if args.save:
                with open(log_path, 'a') as fout:
                    fout.write('{},{}\n'.format(global_step, dev_acc))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                best_dev_epoch = epoch_id
                if args.save:
                    torch.save([model, args], model_path)
                    print(f'model saved to {model_path}')
            model.train()
            start_time = time.time()
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break
            '''
            test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
            print('-' * 71)
            print('| step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(global_step, dev_acc, test_acc))
            print('-' * 71)
            if args.save:
                with open(log_path, 'a') as fout:
                    fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save:
                    torch.save([model, args], model_path)
                    print(f'model saved to {model_path}')
            model.train()
            start_time = time.time()
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break
            '''
    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev acc: {:.4f} (at epoch {})'.format(best_dev_acc, best_dev_epoch))
    #print('final test acc: {:.4f}'.format(final_test_acc))
    print()


def eval(args):
    raise NotImplementedError()


def pred(args):
    raise NotImplementedError()


if __name__ == '__main__':
    main()
