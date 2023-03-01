import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from transformers import ElectraTokenizerFast, ElectraConfig
from transformers import RobertaTokenizerFast, RobertaConfig
from transformers import BertTokenizer, BertConfig
from models import ElectraForSequenceClassification
from models import RobertaForSequenceClassification
#from models import BertForSequenceClassification
from utils import convert_data2tensordataset, retrieve_all_instances_from_dataset, retrieve_labels_from_dataset_for_classification
from src.multi_task.func.make_multitask_dataset_final import retrieve_all_processed_data_from_dataset

import os
from data import write_predictions_to_file
import datetime


def do_train(config, model, optimizer, scheduler, train_dataloader, epoch, global_step):
    losses = []
    total_predicts, total_corrects = [], []
    for step, batch in enumerate(tqdm(train_dataloader, desc='do_train(epoch_{})'.format(epoch))):
        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        loss, predicts = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predicts = predicts.argmax(dim=-1)
        predicts = predicts.cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy().tolist()

        total_predicts.extend(predicts)
        total_corrects.extend(labels)

        if config["gradient_accumulation_steps"] > 1:
            loss = loss / config["gradient_accumulation_steps"]

        loss.backward()
        losses.append(loss.data.item())
        if (step + 1) % config["gradient_accumulation_steps"] == 0 or \
                (len(train_dataloader) <= config["gradient_accumulation_steps"] and (step + 1) == len(
                    train_dataloader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

            # 모델 내부 각 매개변수 가중치 갱신
            optimizer.step()
            scheduler.step()

            # 변화도를 0으로 변경
            model.zero_grad()
            global_step += 1
    accuracy = accuracy_score(total_corrects, total_predicts)
    return accuracy, np.mean(losses), global_step


def do_evaluate(model, test_dataloader):
    total_predicts, total_corrects = [], []
    for step, batch in enumerate(tqdm(test_dataloader, desc='do_evaluate')):
        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        predicts = model(input_ids=input_ids, attention_mask=attention_mask)

        predicts = predicts.argmax(dim=-1)
        predicts = predicts.detach().cpu().tolist()
        labels = labels.detach().cpu().tolist()
        total_predicts.extend(predicts)
        total_corrects.extend(labels)

    accuracy = accuracy_score(total_corrects, total_predicts)
    return accuracy, total_predicts


def train(config):
    # electra_config = ElectraConfig.from_pretrained('google/electra-large-discriminator',
    #                                                num_labels=config['num_labels'],
    #                                                max_length=config['max_length'])
    # electra_tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    # electra_classification = ElectraForSequenceClassification.from_pretrained('google/electra-large-discriminator', config=electra_config)
    #
    # electra_classification.cuda()

    roberta_config = RobertaConfig.from_pretrained('roberta-large', num_labels=config['num_labels'], max_length=config['max_length'])
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    roberta_classification = RobertaForSequenceClassification.from_pretrained('roberta-large', config=roberta_config)
    roberta_classification.cuda()

    # 데이터 로드


    train_set = pd.read_csv(config.path_to_train, sep='\t', quoting=3)
    _, prev_sentences_train, now_sentences_train, next_sentences_train = retrieve_all_instances_from_dataset(train_set)
    training_label_set = pd.read_csv(config.path_to_training_labels, sep="\t", header=None, names=["Id", "Label"])
    label_list_train = retrieve_labels_from_dataset_for_classification(training_label_set)

    train_dataset = convert_data2tensordataset(prev_sentences_train, now_sentences_train, next_sentences_train, label_list_train,
                                               tokenizer=roberta_tokenizer, max_length=config['max_length'])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config['batch_size'])

    dev_set = pd.read_csv(config.path_to_dev, sep='\t', quoting=3)
    _, prev_sentences_dev, now_sentences_dev, next_sentences_dev = retrieve_all_instances_from_dataset(dev_set)
    dev_label_set = pd.read_csv(config.path_to_dev_labels, sep="\t", header=None, names=["Id", "Label"])
    label_list_dev = retrieve_labels_from_dataset_for_classification(dev_label_set)

    dev_dataset = convert_data2tensordataset(prev_sentences_dev, now_sentences_dev, next_sentences_dev, label_list_dev,
                                             tokenizer=roberta_tokenizer, max_length=config['max_length'])
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=config['batch_size'])

    t_total = len(train_dataloader) // config["gradient_accumulation_steps"] * config["epoch"]
    optimizer = AdamW(roberta_classification.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=config["warmup_steps"],
                                                num_training_steps=t_total)

    global_step = 0
    roberta_classification.zero_grad()
    max_test_accuracy = 0
    for epoch in range(config["epoch"]):
        roberta_classification.train()

        # 학습 데이터에 대한 정확도와 평균 loss
        train_accuracy, average_loss, global_step = do_train(config=config, model=roberta_classification,
                                                             optimizer=optimizer, scheduler=scheduler,
                                                             train_dataloader=train_dataloader,
                                                             epoch=epoch + 1, global_step=global_step)

        print("train_accuracy : {}\taverage_loss : {}\n".format(round(train_accuracy, 4), round(average_loss, 4)))

        roberta_classification.eval()

        # 평가 데이터에 대한 정확도
        test_accuracy, _ = do_evaluate(model=roberta_classification, test_dataloader=dev_dataloader)

        print("test_accuracy : {}\n".format(round(test_accuracy, 4)))

        if max_test_accuracy < test_accuracy:
            max_test_accuracy = test_accuracy
            output = os.path.join(config.output_dir, "{}_checkpoint_{}".format(config.PMmodel, epoch))
            if not os.path.exists(output):
                os.makedirs(output)
            roberta_config.save_pretrained(output)
            roberta_tokenizer.save_pretrained(output)
            roberta_classification.save_pretrained(output)
            with open(os.path.join(output, 'README.md'), 'a') as file:
                file.write('test_accuracy: {}\n'.format(test_accuracy))
                file.write('average_loss : {}\n'.format(average_loss))
                file.write('electra_config : {}\n'.format(roberta_config))


# 추후 테스트 데이터셋 나오면 변경
def dev(config):
    electra_config = ElectraConfig.from_pretrained(
        os.path.join(config.output_dir, "{}_checkpoint_{}".format(config.PMmodel, config.checkpoint))
    )
    electra_tokenizer = ElectraTokenizerFast.from_pretrained(
        os.path.join(config.output_dir, "{}_checkpoint_{}".format(config.PMmodel, config.checkpoint))
    )
    electra_classification = ElectraForSequenceClassification.from_pretrained(
        os.path.join(config.output_dir, "{}_checkpoint_{}".format(config.PMmodel, config.checkpoint)),
        config=electra_config
    )
    electra_classification.cuda()

    dev_set = pd.read_csv(config.path_to_dev, sep='\t', quoting=3)
    dev_ids, prev_sentence_dev, now_sentence_dev, next_sentences_dev = retrieve_all_instances_from_dataset(dev_set)
    dev_label_set = pd.read_csv(config.path_to_dev_labels, sep='\t', header=None, names=['Id', 'Label'])
    label_list_dev = retrieve_labels_from_dataset_for_classification(dev_label_set)

    dev_dataset = convert_data2tensordataset(prev_sentence_dev, now_sentence_dev, next_sentences_dev, label_list_dev,
                                             tokenizer=electra_tokenizer, max_length=config.max_length)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=config['batch_size'])

    test_accuracy, total_predicts = do_evaluate(model=electra_classification, test_dataloader=dev_dataloader)
    print("test_accuracy : {}\n".format(round(test_accuracy, 4)))

    prediction_dataframe = write_predictions_to_file(
        path_to_predictions=config.path_to_predictions,
        ids=dev_ids,
        predictions=total_predicts,
        subtask=config.subtask,
    )

def test(config):
    electra_config = ElectraConfig.from_pretrained(
        os.path.join(config.output_dir, "{}_checkpoint_{}".format(config.PMmodel, config.checkpoint))
    )
    electra_tokenizer = ElectraTokenizerFast.from_pretrained(
        os.path.join(config.output_dir, "{}_checkpoint_{}".format(config.PMmodel, config.checkpoint))
    )
    electra_classification = ElectraForSequenceClassification.from_pretrained(
        os.path.join(config.output_dir, "{}_checkpoint_{}".format(config.PMmodel, config.checkpoint)),
        config=electra_config
    )
    electra_classification.cuda()

    test_set = pd.read_csv(config.path_to_test, sep='\t', quoting=3)
    test_ids, prev_sentence_dev, now_sentence_dev, next_sentences_dev = retrieve_all_instances_from_dataset(test_set)
    dev_label_set = pd.read_csv(config.path_to_dev_labels, sep='\t', header=None, names=['Id', 'Label'])
    label_list_dev = retrieve_labels_from_dataset_for_classification(dev_label_set)

    dev_dataset = convert_data2tensordataset(prev_sentence_dev, now_sentence_dev, next_sentences_dev, label_list_dev,
                                             tokenizer=electra_tokenizer, max_length=config.max_length)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=config['batch_size'])

    test_accuracy, total_predicts = do_evaluate(model=electra_classification, test_dataloader=dev_dataloader)
    print("test_accuracy : {}\n".format(round(test_accuracy, 4)))

    prediction_dataframe = write_predictions_to_file(
        path_to_predictions=config.path_to_predictions,
        ids=test_ids,
        predictions=total_predicts,
        subtask=config.subtask,
    )