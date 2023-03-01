from transformers import ElectraConfig, ElectraTokenizerFast
import pickle as pk
import torch
import os
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np

# 여기서 모델 리니어 한줄, 관계만, 관계 유형만, 관계+유형
from src.multi_task.model.model import ElectraForMultiTaskClassification
from src.multi_task.model.use_paper_model import ElectraForClassificationOneLinear, ElectraForClassificationOnlyExist, ElectraForClassificationOnlyType
from src.multi_task.model.use_paper_model import ElectraForClassificationOnlyExistWithDownSample, ElectraForClassificationOnlyTypeWithDownSample, ElectraForClassificationOnlyTypeWithDownSampleNoRelation
from src.multi_task.model.use_paper_model import ElectraForClassificationExistAndType, ElectraForClassificationExistAndTypeWithDownSample, ElectraForClassificationExistAndTypeWithDownSampleNoRelationRemove


def load_pickle_data(path):
    with open(path, 'rb') as f:
        data = pk.load(f)
    total_input_ids = torch.tensor(data['total_input_ids'], dtype=torch.long)
    total_attention_mask = torch.tensor(data['total_attention_mask'], dtype=torch.long)
    total_plausible_label = torch.tensor(data['total_plausible_label'], dtype=torch.long)
    total_pair_ids = torch.tensor(data['total_pair_ids'], dtype=torch.long)
    total_relation_exist_label = torch.tensor(data['total_relation_exist_label'], dtype=torch.long)
    total_relation_type_label = torch.tensor(data['total_relation_type_label'], dtype=torch.long)
    dataset = TensorDataset(total_input_ids, total_attention_mask, total_plausible_label,
                            total_pair_ids, total_relation_exist_label, total_relation_type_label)
    return dataset


def do_train(args, model, optimizer, scheduler, train_dataloader, epoch, global_step):
    losses = []
    total_predicts_plausible, total_corrects_plausible = [], []
    total_predicts_relation_exist, total_corrects_relation_exist = [], []
    total_predicts_relation_type, total_corrects_relation_type = [], []
    for step, batch in enumerate(tqdm(train_dataloader, desc='do_train(epoch_{})'.format(epoch))):
        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask, plausible_label, pair_ids, relation_exist_label, relation_type_label \
            = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        # 리니어 하나일땐 사용 ㄴㄴ
        loss, predict_plausible, predict_relation_exist, predict_relation_type, correct_exist, correct_type = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pair_ids=pair_ids,
            relation_exist_label=relation_exist_label,
            relation_type_label=relation_type_label,
            plausible_label=plausible_label
        )
        # 관계 존재만
        # loss, predict_plausible, predict_relation_exist, correct_relation_exist = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pair_ids=pair_ids,
        #     relation_exist_label=relation_exist_label,
        #     relation_type_label=relation_type_label,
        #     plausible_label=plausible_label
        # )

        # 리니어 하나 일때만 사용
        # loss, predict_plausible = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pair_ids=pair_ids,
        #     relation_exist_label=relation_exist_label,
        #     relation_type_label=relation_type_label,
        #     plausible_label=plausible_label
        # )
        # 관계 유형만 했을 때
        # loss, predict_plausible, predict_relation_type, correct_relation_type = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pair_ids=pair_ids,
        #     relation_exist_label=relation_exist_label,
        #     relation_type_label=relation_type_label,
        #     plausible_label=plausible_label
        # )

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
        correct_relation_exist = correct_exist.cpu().detach().numpy().tolist()
        correct_relation_type = correct_type.cpu().detach().numpy().tolist()
        # correct_relation_exist = downsample_correct_exist.cpu().detach().numpy().tolist()
        # correct_relation_type = downsample_correct_type.cpu().detach().numpy().tolist()

        total_corrects_plausible.extend(correct_plausible)
        total_corrects_relation_exist.extend(correct_relation_exist)
        # total_corrects_relation_exist.extend(correct_relation_exist)
        total_corrects_relation_type.extend(correct_relation_type)
        loss.backward()
        losses.append(loss.data.item())
        if (step + 1) % args.gradient_accumulation_steps == 0 or \
                (len(train_dataloader) <= args.gradient_accumulation_steps and (step + 1) == len(train_dataloader)):
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
    # 한번에 다 할때
    return accuracy_plausible, accuracy_relation_exist, accuracy_relation_type, np.mean(losses), global_step
    # 리니어 한개만 했을 때
    # return accuracy_plausible, np.mean(losses), global_step
    # 관계 유무만 했을 때
    #return accuracy_plausible, accuracy_relation_exist, np.mean(losses), global_step
    # 관계 유형만 했을 때
    #return accuracy_plausible, accuracy_relation_type, np.mean(losses), global_step

def do_evaluate(model, dev_dataloader):
    total_predicts_plausible, total_corrects_plausible = [], []
    total_predicts_relation_exist, total_corrects_relation_exist = [], []
    total_predicts_relation_type, total_corrects_relation_type = [], []
    for step, batch in enumerate(tqdm(dev_dataloader, desc="do evaluate")):
        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask, plausible_label, pair_ids, relation_exist_label, relation_type_label \
            = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

        # loss, logit, logit, logit 존재랑 유형 다 했을때
        predict_plausible, predict_relation_exist, predict_relation_type, correct_exist, correct_type = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pair_ids=pair_ids,
            relation_exist_label=relation_exist_label,
            relation_type_label=relation_type_label
        )
        # 관계 유형만 했을 때
        # predict_plausible, predict_relation_type, correct_type = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pair_ids=pair_ids,
        #     relation_exist_label=relation_exist_label,
        #     relation_type_label=relation_type_label
        # )
        # 관계 존재만 했을 때
        # predict_plausible, predict_relation_exist, correct_exist = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pair_ids=pair_ids,
        #     relation_exist_label=relation_exist_label,
        #     relation_type_label=relation_type_label
        # )
        # 리니어 하나만 했을때 
        # predict_plausible = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pair_ids=pair_ids,
        #     relation_exist_label=relation_exist_label,
        #     relation_type_label=relation_type_label
        # )
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
        correct_relation_exist = correct_exist.cpu().detach().numpy().tolist()
        correct_relation_type = correct_type.cpu().detach().numpy().tolist()
        # correct_relation_exist = downsample_correct_exist.cpu().detach().numpy().tolist()
        # correct_relation_type = downsample_correct_type.cpu().detach().numpy().tolist()

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
    # return accuracy_plausible, accuracy_relation_type
    # 리니어 한개만
    # return accuracy_plausible
    # 관계 유무만
    # return accuracy_plausible, accuracy_relation_exist

def train(args):
    electra_config = ElectraConfig.from_pretrained(
        'google/electra-base-discriminator',
        num_labels=args.plausible_num_label,
        max_length=args.input_sentence_max_length
    )
    setattr(electra_config, 'num_relation_exist', args.relation_exist_label)
    setattr(electra_config, 'num_relation_type', args.relation_type_label)
    electra_tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')
    electra_multi_task_classification = ElectraForClassificationExistAndTypeWithDownSampleNoRelationRemove.from_pretrained(
        'google/electra-base-discriminator', config=electra_config)
    electra_multi_task_classification.cuda()

    train_dataset = load_pickle_data('../../data/save_pickle_train_data.pkl')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    dev_dataset = load_pickle_data('../../data/save_pickle_dev_data.pkl')
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
        # 다할때
        train_accuracy_plausible, train_accuracy_relation_exist, train_accuracy_relation_type, average_loss, global_step = do_train(
            args=args, model=electra_multi_task_classification,
            optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader,
            epoch=epoch + 1, global_step=global_step
        )
        # 리니어 하나만 했을 때
        # train_accuracy_plausible, average_loss, global_step = do_train(
        #     args=args, model=electra_multi_task_classification,
        #     optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader,
        #     epoch=epoch + 1, global_step=global_step
        # )
        # 관계만 했을 때
        # train_accuracy_plausible, train_accuracy_relation_exist, average_loss, global_step = do_train(
        #     args=args, model=electra_multi_task_classification,
        #     optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader,
        #     epoch=epoch + 1, global_step=global_step
        # )
        # 관계 유형만 했을 때
        # train_accuracy_plausible, train_accuracy_relation_type, average_loss, global_step = do_train(
        #     args=args, model=electra_multi_task_classification,
        #     optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader,
        #     epoch=epoch + 1, global_step=global_step
        # )
        print("\n"
              "train_accuracy_plausible : {}\n"
              "train_accuracy_relation_exist : {}\n"
              "train_accuracy_relation_type : {}\n"
              "average_loss : {}\n".format(round(train_accuracy_plausible, 4), round(train_accuracy_relation_exist, 4), round(train_accuracy_relation_type, 4), round(average_loss, 4)))
        # 리니어 하나만
        # print("\n"
        #       "train_accuracy_plausible : {}\n"
        #       "average_loss : {}\n".format(round(train_accuracy_plausible, 4), round(average_loss, 4)))
        # 관계 유형만 했을 때
        # print("\n"
        #       "train_accuracy_plausible : {}\n"
        #       "train_accuracy_relation_type : {}\n"
        #       "average_loss : {}\n".format(round(train_accuracy_plausible, 4), round(train_accuracy_relation_type, 4), round(average_loss, 4)))

        # 관계 존재만 했을 때
        # print("\n"
        #       "train_accuracy_plausible : {}\n"
        #       "train_accuracy_relation_exist : {}\n"
        #       "average_loss : {}\n".format(round(train_accuracy_plausible, 4), round(train_accuracy_relation_exist, 4), round(average_loss, 4)))

        electra_multi_task_classification.eval()
        # 다 할 때
        dev_accuracy_plausible, dev_accuracy_relation_exist, dev_accuracy_relation_type = do_evaluate(model=electra_multi_task_classification, dev_dataloader=dev_dataloader)
        # 존재만 했을 떄
        # dev_accuracy_plausible, dev_accuracy_relation_exist = do_evaluate(model=electra_multi_task_classification, dev_dataloader=dev_dataloader)
        # 리니어 하나만
        #dev_accuracy_plausible = do_evaluate(model=electra_multi_task_classification, dev_dataloader=dev_dataloader)
        # 관계 유형만 했을 ㄸ ㅐ
        # dev_accuracy_plausible, dev_accuracy_relation_type = do_evaluate(model=electra_multi_task_classification, dev_dataloader=dev_dataloader)
        # 두가지 다 할 때
        print("\n"
              "dev_accuracy_plausible : {}\n"
              "dev_accuracy_relation_exist : {}\n"
              "dev_accuracy_relation_type : {}\n".format(round(dev_accuracy_plausible, 4), round(dev_accuracy_relation_exist, 4), round(dev_accuracy_relation_type, 4)))
        # 관계존재만 했을 때
        # print("\n"
        #       "dev_accuracy_plausible : {}\n"
        #       "dev_accuracy_relation_exist : {}\n"
        #       .format(round(dev_accuracy_plausible, 4), round(dev_accuracy_relation_exist, 4)))
        # 리니어 한개
        # print("\n"
        #       "dev_accuracy_plausible : {}\n".format(round(dev_accuracy_plausible, 4)))
        # 관계 유형만
        # print("\n"
        #       "dev_accuracy_plausible : {}\n"
        #       "dev_accuracy_relation_type : {}\n".format(round(dev_accuracy_plausible, 4), round(dev_accuracy_relation_type, 4)))
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
                # file.write('dev_accuracy_relation_exist : {}\n'.format(dev_accuracy_relation_exist))
                # file.write('dev_accuracy_relation_type : {}\n'.format(dev_accuracy_relation_type))


