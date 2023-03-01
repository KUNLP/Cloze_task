import json
import spacy
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import ElectraTokenizerFast, ElectraConfig
from transformers import BertTokenizer, BertConfig
from src.multi_task.model.model import BertForMultiTaskClassification

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd

from src.multi_task.model.model import ElectraForMultiTaskClassification
from src.multi_task.func.my_utils import retrieve_all_instances_from_dataset, retrieve_labels_from_dataset_for_classification
from src.multi_task.func.grounding_concept import read_concept_vocab, load_matcher
from src.multi_task.func.multitask_utils_final import read_my_multi_dataset, convert_multidata2tensordataset, convert_multitask_dataset_dev_to_tensordataset, convert_all_instance_to_tensordataset
from src.multi_task.func.multitask_utils_final import load_pickle_data, convert_all_instance_to_tensordataset_test
from src.multi_task.func.make_multitask_dataset_final import retrieve_all_processed_data_from_dataset

from src.data import write_predictions_to_file
from src.use_kagnet.modeling_rgcn import LMRGCN
import wandb

def do_train(args, model, optimizer, scheduler, train_dataloader, epoch, global_step):
    losses = []
    total_predicts_plausible, total_corrects_plausible = [], []
    total_predicts_relation_exist, total_corrects_relation_exist = [], []
    total_predicts_relation_type, total_corrects_relation_type = [], []
    for step, batch in enumerate(tqdm(train_dataloader, desc='do_train(epoch_{})'.format(epoch))):
        batch = tuple(t.cuda() for t in batch)
        #input_ids, attention_mask, concept_pair_ids, relation_exist, relation_type, plausible_label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        input_ids, attention_mask, plausible_label, pair_ids, relation_exist_label, relation_type_label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        # loss, logit, logit, logit
        # loss, predict_plausible, predict_relation_exist, predict_relation_type = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pair_ids=pair_ids,
        #     relation_exist_label=relation_exist_label,
        #     relation_type_label=relation_type_label,
        #     plausible_label=plausible_label
        # )

        loss, predict_plausible, predict_relation_exist, predict_relation_type, downsample_correct_exist, downsample_correct_type = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pair_ids=pair_ids,
            relation_exist_label=relation_exist_label,
            relation_type_label=relation_type_label,
            plausible_label=plausible_label
        )

        # predict
        predict_plausible = predict_plausible.argmax(dim=-1)
        predict_relation_exist = predict_relation_exist.argmax(dim=-1)
        predict_relation_type = predict_relation_type.argmax(dim=-1)
        #predict_relation_exist = [predict.argmax(dim=-1).squeeze() for predict in predict_relation_exist]
        #predict_relation_type = [predict.argmax(dim=-1).squeeze() for predict in predict_relation_type]

        predict_plausible = predict_plausible.cpu().detach().numpy().tolist()
        predict_relation_exist = predict_relation_exist.cpu().detach().numpy().tolist()
        predict_relation_type = predict_relation_type.cpu().detach().numpy().tolist()
        # predict_relation_exist 아예 리스트 하나로 쭉 편 모양으로 하자
        '''
        predict_relation_exist = [predict_list.cpu().detach().numpy().tolist() for predict_list in predict_relation_exist]
        predict_relation_exist = [exist for sublist in predict_relation_exist for exist in sublist]
        predict_relation_type = [predict_list.cpu().detach().numpy().tolist() for predict_list in predict_relation_type]
        predict_relation_type = [item for sublist in predict_relation_type for item in sublist]
        '''
        total_predicts_plausible.extend(predict_plausible)
        total_predicts_relation_exist.extend(predict_relation_exist)
        total_predicts_relation_type.extend(predict_relation_type)

        # correct
        correct_plausible = plausible_label.cpu().detach().numpy().tolist()
        correct_relation_exist = downsample_correct_exist.cpu().detach().numpy().tolist()
        correct_relation_type = downsample_correct_type.cpu().detach().numpy().tolist()
        # correct_relation_exist = []
        # for exist_list in relation_exist_label.cpu().detach().numpy().tolist():
        #     for exist_label in exist_list:
        #         if exist_label != 2:
        #             correct_relation_exist.append(exist_label)
        #         else:
        #             break


        #correct_relation_exist = [exist_label for exist_label in exist_list for exist_list in  if exist_label != 2]
        # correct_relation_type = []
        # for type_list in relation_type_label.cpu().detach().numpy().tolist():
        #     for type_label in type_list:
        #         if type_label != 18:
        #             correct_relation_type.append(type_label)
        #         else:
        #             break

        total_corrects_plausible.extend(correct_plausible)
        total_corrects_relation_exist.extend(correct_relation_exist)
        total_corrects_relation_type.extend(correct_relation_type)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        losses.append(loss.data.item())
        if (step + 1) % args.gradient_accumulation_steps == 0 or \
                (len(train_dataloader) <= args.gradient_accumulation_steps and (step + 1) == len(
                    train_dataloader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 모델 내부 각 매개변수 가중치 갱신
            optimizer.step()
            scheduler.step()

            # 변화도를 0으로 변경
            model.zero_grad()
            global_step += 1
    accuracy_plausible = accuracy_score(total_corrects_plausible, total_predicts_plausible)
    accuracy_relation_exist = accuracy_score(total_corrects_relation_exist, total_predicts_relation_exist)
    accuracy_relation_type = accuracy_score(total_corrects_relation_type, total_predicts_relation_type)
    #print(losses)
    print("plausible macro f1 : {}\t plausible micro f1 : {}\t".format(
        round(f1_score(total_corrects_plausible, total_predicts_plausible, average='macro'), 4),
        round(f1_score(total_corrects_plausible, total_predicts_plausible, average='micro'), 4)
    ))
    print("relation exist macro f1 : {}\t relation exist micro f1 : {}\t".format(
        round(f1_score(total_corrects_relation_exist, total_predicts_relation_exist, average='macro'), 4),
        round(f1_score(total_corrects_relation_exist, total_predicts_relation_exist, average='micro'), 4)
    ))
    print("relation type macro f1 : {}\t relation type micro f1 : {}\t".format(
        round(f1_score(total_corrects_relation_type, total_predicts_relation_type, average='macro'), 4),
        round(f1_score(total_corrects_relation_type, total_predicts_relation_type, average='micro'), 4)
    ))

    return accuracy_plausible, accuracy_relation_exist, accuracy_relation_type, np.mean(losses), global_step


def do_evaluate(model, dev_dataloader):
    total_predicts_plausible, total_corrects_plausible = [], []
    total_predicts_relation_exist, total_corrects_relation_exist = [], []
    total_predicts_relation_type, total_corrects_relation_type = [], []
    for step, batch in enumerate(tqdm(dev_dataloader, desc="do evaluate")):
        batch = tuple(t.cuda() for t in batch)
        # input_ids, attention_mask, concept_pair_ids, relation_exist, relation_type, plausible_label = batch[0], batch[
        #     1], batch[2], batch[3], batch[4], batch[5]
        input_ids, attention_mask, plausible_label, pair_ids, relation_exist_label, relation_type_label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

        # loss, logit, logit, logit
        predict_plausible, predict_relation_exist, predict_relation_type, downsample_correct_exist, downsample_correct_type = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pair_ids=pair_ids,
            relation_exist_label=relation_exist_label,
            relation_type_label=relation_type_label
        )
        # predict
        predict_plausible = predict_plausible.argmax(dim=-1)
        predict_relation_exist = predict_relation_exist.argmax(dim=-1)
        predict_relation_type = predict_relation_type.argmax(dim=-1)
        predict_plausible = predict_plausible.cpu().detach().numpy().tolist()
        predict_relation_exist = predict_relation_exist.cpu().detach().numpy().tolist()
        predict_relation_type = predict_relation_type.cpu().detach().numpy().tolist()

        total_predicts_plausible.extend(predict_plausible)
        total_predicts_relation_exist.extend(predict_relation_exist)
        total_predicts_relation_type.extend(predict_relation_type)

        # correct
        correct_plausible = plausible_label.cpu().detach().numpy().tolist()
        correct_relation_exist = downsample_correct_exist.cpu().detach().numpy().tolist()
        correct_relation_type = downsample_correct_type.cpu().detach().numpy().tolist()
        # correct_relation_exist = []
        # for exist_list in relation_exist_label.cpu().detach().numpy().tolist():
        #     for exist_label in exist_list:
        #         if exist_label != 2:
        #             correct_relation_exist.append(exist_label)
        #         else:
        #             break
        #
        # # correct_relation_exist = [exist_label for exist_label in exist_list for exist_list in  if exist_label != 2]
        # correct_relation_type = []
        # for type_list in relation_type_label.cpu().detach().numpy().tolist():
        #     for type_label in type_list:
        #         if type_label != 18:
        #             correct_relation_type.append(type_label)
        #         else:
        #             break

        total_corrects_plausible.extend(correct_plausible)
        total_corrects_relation_exist.extend(correct_relation_exist)
        total_corrects_relation_type.extend(correct_relation_type)
    accuracy_plausible = accuracy_score(total_corrects_plausible, total_predicts_plausible)
    accuracy_relation_exist = accuracy_score(total_corrects_relation_exist, total_predicts_relation_exist)
    accuracy_relation_type = accuracy_score(total_corrects_relation_type, total_predicts_relation_type)

    print("plausible macro f1 : {}\t plausible micro f1 : {}\t".format(
        round(f1_score(total_corrects_plausible, total_predicts_plausible, average='macro'), 4),
        round(f1_score(total_corrects_plausible, total_predicts_plausible, average='micro'), 4)
    ))
    print("relation exist macro f1 : {}\t relation exist micro f1 : {}\t".format(
        round(f1_score(total_corrects_relation_exist, total_predicts_relation_exist, average='macro'), 4),
        round(f1_score(total_corrects_relation_exist, total_predicts_relation_exist, average='micro'), 4)
    ))
    print("relation type macro f1 : {}\t relation type micro f1 : {}\t".format(
        round(f1_score(total_corrects_relation_type, total_predicts_relation_type, average='macro'), 4),
        round(f1_score(total_corrects_relation_type, total_predicts_relation_type, average='micro'), 4)
    ))
    return accuracy_plausible, accuracy_relation_exist, accuracy_relation_type


def do_test(model, test_dataloader):
    total_predicts_plausible = []
    total_predicts_relation_exist, total_corrects_relation_exist = [], []
    total_predicts_relation_type, total_corrects_relation_type = [], []
    for step, batch in enumerate(tqdm(test_dataloader, desc="do evaluate")):
        batch = tuple(t.cuda() for t in batch)
        # input_ids, attention_mask, concept_pair_ids, relation_exist, relation_type, plausible_label = batch[0], batch[
        #     1], batch[2], batch[3], batch[4], batch[5]
        input_ids, attention_mask, pair_ids, relation_exist_label, relation_type_label = batch[0], batch[1], batch[2], batch[3], batch[4]

        # loss, logit, logit, logit
        predict_plausible, predict_relation_exist, predict_relation_type, downsample_correct_exist, downsample_correct_type = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pair_ids=pair_ids,
            relation_exist_label=relation_exist_label,
            relation_type_label=relation_type_label
        )
        # predict
        predict_plausible = predict_plausible.argmax(dim=-1)
        predict_relation_exist = predict_relation_exist.argmax(dim=-1)
        predict_relation_type = predict_relation_type.argmax(dim=-1)
        predict_plausible = predict_plausible.cpu().detach().numpy().tolist()
        predict_relation_exist = predict_relation_exist.cpu().detach().numpy().tolist()
        predict_relation_type = predict_relation_type.cpu().detach().numpy().tolist()

        total_predicts_plausible.extend(predict_plausible)
        total_predicts_relation_exist.extend(predict_relation_exist)
        total_predicts_relation_type.extend(predict_relation_type)

        # correct
        correct_relation_exist = downsample_correct_exist.cpu().detach().numpy().tolist()
        correct_relation_type = downsample_correct_type.cpu().detach().numpy().tolist()

        total_corrects_relation_exist.extend(correct_relation_exist)
        total_corrects_relation_type.extend(correct_relation_type)
    #accuracy_plausible = accuracy_score(total_corrects_plausible, total_predicts_plausible)
    accuracy_relation_exist = accuracy_score(total_corrects_relation_exist, total_predicts_relation_exist)
    accuracy_relation_type = accuracy_score(total_corrects_relation_type, total_predicts_relation_type)

    print("relation exist macro f1 : {}\t relation exist micro f1 : {}\t".format(
        round(f1_score(total_corrects_relation_exist, total_predicts_relation_exist, average='macro'), 4),
        round(f1_score(total_corrects_relation_exist, total_predicts_relation_exist, average='micro'), 4)
    ))
    print("relation type macro f1 : {}\t relation type micro f1 : {}\t".format(
        round(f1_score(total_corrects_relation_type, total_predicts_relation_type, average='macro'), 4),
        round(f1_score(total_corrects_relation_type, total_predicts_relation_type, average='micro'), 4)
    ))
    return total_predicts_plausible, accuracy_relation_exist, accuracy_relation_type


def train(args):
    electra_config = ElectraConfig.from_pretrained(
        'google/electra-large-discriminator',
        num_labels=args.plausible_num_label,
        max_length=args.input_sentence_max_length
    )
    setattr(electra_config, 'num_relation_exist', args.relation_exist_label)
    setattr(electra_config, 'num_relation_type', args.relation_type_label)
    electra_tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    electra_multi_task_classification = ElectraForMultiTaskClassification.from_pretrained(
        'google/electra-large-discriminator', config=electra_config)
    electra_multi_task_classification.cuda()
    wandb.watch(electra_multi_task_classification)

    # 원본 train 데이터셋 로딩
    # train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    # _, article_title, prev_sentences_train, now_sentences_train, next_sentences_train, answers_train = retrieve_all_instances_from_dataset(train_set)
    # train_label_set = pd.read_csv(args.path_to_train_labels, sep='\t', header=None, names=["Id", "Label"])
    # train_plausible_labels = retrieve_labels_from_dataset_for_classification(train_label_set)

    train_set = pd.read_csv(args.path_to_train, sep='\t', quoting=3)
    ids, train_titles, train_processed_sentences, train_answers, train_original_answer_char_positions, \
    train_prev_sentences, train_now_sentences, train_next_sentences = retrieve_all_processed_data_from_dataset(
        train_set)
    train_label_set = pd.read_csv(args.path_to_train_labels, sep='\t', header=None, names=["Id", "Label"])
    train_plausible_labels = retrieve_labels_from_dataset_for_classification(train_label_set)

    # 원본 dev 데이터셋 로딩
    # dev_set = pd.read_csv(args.path_to_dev, sep='\t', quoting=3)
    # _, article_title, prev_sentences_dev, now_sentences_dev, next_sentences_dev, answers_dev = retrieve_all_instances_from_dataset(dev_set)
    # dev_label_set = pd.read_csv(args.path_to_dev_labels, sep='\t', header=None, names=["Id", "Label"])
    # dev_plausible_labels = retrieve_labels_from_dataset_for_classification(dev_label_set)

    dev_set = pd.read_csv(args.path_to_dev, sep='\t', quoting=3)
    ids, dev_titles, dev_processed_sentences, dev_answers, dev_original_answer_char_positions, \
    dev_prev_sentences, dev_now_sentences, dev_next_sentences = retrieve_all_processed_data_from_dataset(
        dev_set)
    dev_label_set = pd.read_csv(args.path_to_dev_labels, sep='\t', header=None, names=['Id', 'Label'])
    dev_plausible_labels = retrieve_labels_from_dataset_for_classification(dev_label_set)

    # 컨셉넷 관련 로딩
    #conceptnet_vocab = read_concept_vocab(args)
    conceptnet_dataset = pd.read_csv(args.path_to_conceptnet, sep='\t', header=None,
                                     names=['relation', 'header', 'tail', 'weight'])
    # conceptnet_relation = sorted(conceptnet_dataset['relation'].unique().tolist())
    # conceptnet_relation.insert(0, 'no relation')
    # relation2idx = {}
    # idx2relation = {}
    # for idx, relation in enumerate(conceptnet_relation):
    #     relation2idx[relation] = idx
    #     idx2relation[idx] = relation

    # 내가 만든 멀티태스크 데이터셋 로딩
    with open(args.path_to_multitask_dataset_dev, 'r', encoding='utf8') as f:
        multi_task_dataset_dev = json.load(f)
    with open(args.path_to_multitask_dataset_train, 'r', encoding='utf8') as f:
        multi_task_dataset_train = json.load(f)

    # 멀티태스크 데이터셋에서 각각 컨셉틀 추출

    train_sentence_span_concepts = []
    train_answer_span_concepts = []
    for data in multi_task_dataset_train:
        train_sentence_span_concepts.append(data['sentence_span_concept'])
        train_answer_span_concepts.append(data['answer_span_concept'])
    for idx, (train_answer_span_concept, answer) in enumerate(zip(train_answer_span_concepts, train_answers)):
        if not train_answer_span_concept:
            if len(answer.split(" "))> 1:
                train_answer_span_concepts[idx] = [answer.split(" ")[1].lower()]
            else:
                train_answer_span_concepts[idx] = [answer.lower()]
        else:
            train_answer_span_concepts[idx] = [answer_span.lower() for answer_span in train_answer_span_concept]

    dev_sentence_span_concepts = []
    dev_answer_span_concepts = []
    for data in multi_task_dataset_dev:
        dev_sentence_span_concepts.append(data['sentence_span_concept'])
        dev_answer_span_concepts.append(data['answer_span_concept'])
    for idx, (dev_answer_span_concept, answer) in enumerate(zip(dev_answer_span_concepts, dev_answers)):
        if not dev_answer_span_concept:
            if len(answer.split(" "))>1:
                dev_answer_span_concepts[idx] = [answer.split(" ")[1].lower()]
            else:
                dev_answer_span_concepts[idx] = [answer.lower()]
        else:
            dev_answer_span_concepts[idx] = [answer_span.lower() for answer_span in dev_answer_span_concept]


    #train dataset / sampler / dataloader
    # train_dataset = convert_all_instance_to_tensordataset(
    #     args, electra_tokenizer, conceptnet_dataset,
    #     train_titles, train_processed_sentences, train_answers,
    #     train_sentence_span_concepts, train_answer_span_concepts, train_plausible_labels, make_train=True
    # )
    # 이부분에서 피클 로드하는데 전부 트레인 데이터셋만 가져오는거같다.
    train_dataset = load_pickle_data(args, 'train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    #dev dataset / sampler / dataloader
    # dev_dataset = convert_all_instance_to_tensordataset(
    #     args, electra_tokenizer, conceptnet_dataset,
    #     dev_titles, dev_processed_sentences, dev_answers,
    #     dev_sentence_span_concepts, dev_answer_span_concepts, dev_plausible_labels, make_train=False
    # )
    dev_dataset = load_pickle_data(args, 'dev')
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epoch
    optimizer = AdamW(electra_multi_task_classification.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    global_step = 0
    electra_multi_task_classification.zero_grad()
    max_plausible_accuracy = 0
    for epoch in range(args.epoch):
        electra_multi_task_classification.train()
        train_accuracy_plausible, train_accuracy_relation_exist, train_accuracy_relation_type, average_loss, global_step = do_train(
            args=args, model=electra_multi_task_classification,
            optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader,
            epoch=epoch + 1, global_step=global_step
        )

        print("train_accuracy_plausible : {}\n"
              "train_accuracy_relation_exist : {}\n"
              "train_accuracy_relation_type : {}\n"
              "average_loss : {}\n".format(round(train_accuracy_plausible, 4), round(train_accuracy_relation_exist, 4), round(train_accuracy_relation_type, 4), round(average_loss, 4)))

        electra_multi_task_classification.eval()
        dev_accuracy_plausible, dev_accuracy_relation_exist, dev_accuracy_relation_type = do_evaluate(model=electra_multi_task_classification, dev_dataloader=dev_dataloader)
        print("dev_accuracy_plausible : {}\n"
              "dev_accuracy_relation_exist : {}\n"
              "dev_accuracy_relation_type : {}\n".format(round(dev_accuracy_plausible, 4), round(dev_accuracy_relation_exist, 4), round(dev_accuracy_relation_type, 4)))

        if max_plausible_accuracy < dev_accuracy_plausible:
            max_plausible_accuracy = dev_accuracy_plausible
            output = os.path.join(args.output_dir, "{}_checkpoint_{}".format(args.PLM, epoch))
            if not os.path.exists(output):
                os.makedirs(output)
            electra_config.save_pretrained(output)
            electra_tokenizer.save_pretrained(output)
            electra_multi_task_classification.save_pretrained(output)
            with open(os.path.join(output, "README_experiment_result.md"), 'a') as file:
                file.write('dev_accuracy_plausible : {}\n'.format(dev_accuracy_plausible))
                file.write('dev_accuracy_relation_exist : {}\n'.format(dev_accuracy_relation_exist))
                file.write('dev_accuracy_relation_type : {}\n'.format(dev_accuracy_relation_type))


def test(args):
    electra_config = ElectraConfig.from_pretrained(args.load_my_sota)
    electra_tokenizer = ElectraTokenizerFast.from_pretrained(args.load_my_sota)
    electra_multi_task_classification = ElectraForMultiTaskClassification.from_pretrained(args.load_my_sota, config=electra_config)
    electra_multi_task_classification.cuda()

    conceptnet_dataset = pd.read_csv(args.path_to_conceptnet, sep='\t', header=None,
                                     names=['relation', 'header', 'tail', 'weight'])

    test_set = pd.read_csv(args.path_to_test, sep='\t', quoting=3)
    ids, test_titles, test_processed_sentences, test_answers, test_original_answer_char_positions,\
        test_prev_sentences, test_now_sentences, test_next_sentences = retrieve_all_processed_data_from_dataset(test_set)

    with open(args.path_to_multitask_dataset_test, 'r', encoding='utf8') as f:
        multi_task_dataset_test = json.load(f)

    test_sentence_span_concepts = []
    test_answer_span_concepts = []
    for data in multi_task_dataset_test:
        test_sentence_span_concepts.append(data['sentence_span_concept'])
        test_answer_span_concepts.append(data['answer_span_concept'])
    for idx, (test_answer_span_concept, answer) in enumerate(zip(test_answer_span_concepts, test_answers)):
        if not test_answer_span_concept:
            if len(answer.split(" ")) > 1:
                test_answer_span_concepts[idx] = [answer.split(" ")[1].lower()]
            else:
                test_answer_span_concepts[idx] = [answer.lower()]
        else:
            test_answer_span_concepts[idx] = [answer_span.lower() for answer_span in test_answer_span_concept]

    # test_dataset = convert_all_instance_to_tensordataset_test(
    #     args=args, tokenizer=electra_tokenizer, conceptnet_dataset=conceptnet_dataset,
    #     titles_instances=test_titles, processed_sentence_instances=test_processed_sentences,
    #     answer_instances=test_answers,
    #     sentence_concepts=test_sentence_span_concepts, answer_concepts=test_answer_span_concepts
    # )
    # tensor 만드는곳
    test_dataset = load_pickle_data(args)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    total_predicts, test_accuracy_relation_exist, test_accuracy_relation_type = do_test(
        model=electra_multi_task_classification, test_dataloader=test_dataloader
    )
    print(
        "Good luck..."
        "test_accuracy_relation_exist : {}\n"
        "test_accuracy_relation_type : {}\n".format(round(test_accuracy_relation_exist, 4), round(test_accuracy_relation_type, 4))
    )
    prediction_dataframe = write_predictions_to_file(
        path_to_predictions=args.path_to_predictions,
        ids=ids,
        predictions=total_predicts,
        subtask=args.subtask,
    )


def test_test(args):
    bert_config = BertConfig.from_pretrained(args.rcgn_model_path)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    LMRGCN = torch.load(args.rcgn_model_path)
    #bert_multi_task_classification = BertForMultiTaskClassification.from_pretrained(args.rcgn_model_path, config=bFert_config)

    conceptnet_dataset = pd.read_csv(args.path_to_conceptnet, sep='\t', header=None,
                                     names=['relation', 'header', 'tail', 'weight'])

    test_set = pd.read_csv(args.path_to_test, sep='\t', quoting=3)
    ids, test_titles, test_processed_sentences, test_answers, test_original_answer_char_positions,\
        test_prev_sentences, test_now_sentences, test_next_sentences = retrieve_all_processed_data_from_dataset(test_set)

    with open(args.path_to_multitask_dataset_test, 'r', encoding='utf8') as f:
        multi_task_dataset_test = json.load(f)

    test_sentence_span_concepts = []
    test_answer_span_concepts = []
    for data in multi_task_dataset_test:
        test_sentence_span_concepts.append(data['sentence_span_concept'])
        test_answer_span_concepts.append(data['answer_span_concept'])
    for idx, (test_answer_span_concept, answer) in enumerate(zip(test_answer_span_concepts, test_answers)):
        if not test_answer_span_concept:
            if len(answer.split(" ")) > 1:
                test_answer_span_concepts[idx] = [answer.split(" ")[1].lower()]
            else:
                test_answer_span_concepts[idx] = [answer.lower()]
        else:
            test_answer_span_concepts[idx] = [answer_span.lower() for answer_span in test_answer_span_concept]
    test_dataset = convert_all_instance_to_tensordataset_test(
        args=args, tokenizer=bert_tokenizer, conceptnet_dataset=conceptnet_dataset,
        titles_instances=test_titles, processed_sentence_instances=test_processed_sentences,
        answer_instances=test_answers,
        sentence_concepts=test_sentence_span_concepts, answer_concepts=test_answer_span_concepts
    )

    # tensor 만드는곳
    #test_dataset = load_pickle_data(args)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    total_predicts, test_accuracy_relation_exist, test_accuracy_relation_type = do_test(
        model=LMRGCN, test_dataloader=test_dataloader
    )
    print(
        "Good luck..."
        "test_accuracy_relation_exist : {}\n"
        "test_accuracy_relation_type : {}\n".format(round(test_accuracy_relation_exist, 4), round(test_accuracy_relation_type, 4))
    )
    prediction_dataframe = write_predictions_to_file(
        path_to_predictions=args.path_to_predictions,
        ids=ids,
        predictions=total_predicts,
        subtask=args.subtask,
    )



def dev(args):
    return None
