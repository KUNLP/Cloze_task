from transformers import ElectraModel, ElectraPreTrainedModel
import torch.nn as nn
import torch
import pandas as pd
from random import shuffle


#그냥 리니어 하나
class ElectraForClassificationOneLinear(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_label = config.num_labels
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        # 레이어 하나
        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_label)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, plausible_label=None, pair_ids=None, relation_exist_label=None, relation_type_label=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        plausible_logit = self.plausible_layer(outputs[:, 0, :])
        plausible_loss_fct = nn.CrossEntropyLoss()
        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_label), plausible_label.view(-1))
            return plausible_loss, self.softmax(plausible_logit)
        else:
            return self.softmax(plausible_logit)



# 관계 존재만
class ElectraForClassificationOnlyExist(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_label = config.num_labels
        self.num_relation_exist_label = config.num_relation_exist
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        # 레이어 하나
        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_label)
        )
        self.relation_exist_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist_label)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    # 다운샘플링 없이
    def forward(self, input_ids=None, attention_mask=None, plausible_label=None, pair_ids=None, relation_exist_label=None, relation_type_label=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        padding = torch.tensor([[-1,-1], [-1, -1]]).cuda()
        padding_exist = torch.tensor(2).cuda()

        sentence_concept_tensor, answer_concept_tensor = [], []
        exist_label = []
        for batch_idx, (pair_batch_line, exist_batch_line) in enumerate(zip(pair_ids, relation_exist_label)):
            pairbatch_line = [content for content in pair_batch_line if not torch.equal(content, padding)]

            valid_relation_exist_index = torch.stack([content for content in exist_batch_line if not torch.equal(content, padding_exist)], dim=0)
            valid_index = [index for index in range(valid_relation_exist_index.size(dim=0))]
            pairbatch_line = [pairbatch_line[index] for index in valid_index]
            exist_label.extend([valid_relation_exist_index[index] for index in valid_index])
            for pair_line in pairbatch_line:
                if not torch.equal(pair_line, padding):
                    if outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) == 1:
                        sentence_concept_tensor.append(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :])
                    elif outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) != 1:
                        sentence_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :], dim=0, keepdim=True))
                    if outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) == 1:
                        answer_concept_tensor.append(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :])
                    elif outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) != 1:
                        answer_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :], dim=0, keepdim=True))
        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sentence_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(answer_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)

        plausible_loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.CrossEntropyLoss()

        exist_label = torch.stack(exist_label, dim=0)
        exist_logit = self.relation_exist_layer(pair_data_input)


        plausible_logit = self.plausible_layer(outputs[:, 0, :])
        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_label), plausible_label.view(-1))
            exist_loss = exist_loss_fct(exist_logit.view(-1, self.num_relation_exist_label), exist_label.view(-1))
            total_loss = plausible_loss + exist_loss
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), exist_label
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), exist_label


# 관계 존재만
class ElectraForClassificationOnlyExistWithDownSample(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_label = config.num_labels
        self.num_relation_exist_label = config.num_relation_exist
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        # 레이어 하나
        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_label)
        )
        self.relation_exist_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist_label)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_ids=None, attention_mask=None, plausible_label=None, pair_ids=None, relation_exist_label=None, relation_type_label=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        padding = torch.tensor([[-1,-1], [-1, -1]]).cuda()
        padding_exist = torch.tensor(2).cuda()

        sentence_concept_tensor, answer_concept_tensor = [], []
        downsample_exist_label = []

        for batch_idx, (pair_batch_line, exist_batch_line) in enumerate(zip(pair_ids, relation_exist_label)):
            pairbatch_line = [content for content in pair_batch_line if not torch.equal(content, padding)]

            valid_relation_exist_index = torch.stack([content for content in exist_batch_line if not torch.equal(content, padding_exist)], dim=0)
            valid_index = [index for index in range(valid_relation_exist_index.size(dim=0))]
            find_index = [index for index, value in enumerate(valid_relation_exist_index) if not torch.equal(value, torch.tensor(0).cuda())]
            remove_index = list(set(valid_index)-set(find_index))
            shuffle(remove_index)
            downsample = remove_index[:4*len(find_index)] + find_index
            pairbatch_line = [pairbatch_line[index] for index in downsample]
            downsample_exist_label.extend([valid_relation_exist_index[index] for index in downsample])

            for pair_line in pairbatch_line:
                if not torch.equal(pair_line, padding):
                    if outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) == 1:
                        sentence_concept_tensor.append(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :])
                    elif outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) != 1:
                        sentence_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :], dim=0, keepdim=True))
                    if outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) == 1:
                        answer_concept_tensor.append(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :])
                    elif outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) != 1:
                        answer_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :], dim=0, keepdim=True))
        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sentence_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(answer_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)

        plausible_loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.CrossEntropyLoss()

        downsample_exist_label = torch.stack(downsample_exist_label, dim=0)

        exist_logit = self.relation_exist_layer(pair_data_input)
        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_label), plausible_label.view(-1))
            exist_loss = exist_loss_fct(exist_logit.view(-1, self.num_relation_exist_label), downsample_exist_label.view(-1))
            total_loss = plausible_loss + exist_loss
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), downsample_exist_label
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), downsample_exist_label

# 관계 유형만
class ElectraForClassificationOnlyType(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_labels = config.num_labels
        self.num_relation_type_label = config.num_relation_type
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        # 레이어 하나
        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_labels)
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, plausible_label=None, pair_ids=None, relation_exist_label=None, relation_type_label=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        padding = torch.tensor([[-1, -1], [-1, -1]]).cuda()
        padding_type = torch.tensor(18).cuda()

        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        sentence_concept_tensor = []
        answer_concept_tensor = []

        type_label = []
        for batch_idx, (pair_batch_line, type_batch_line) in enumerate(zip(pair_ids, relation_type_label)):
            pairbatch_line = [content for content in pair_batch_line if not torch.equal(content, padding)]

            valid_relation_type_index = torch.stack([content for content in type_batch_line if not torch.equal(content, padding_type)], dim=0)
            valid_index = [index for index in range(valid_relation_type_index.size(dim=0))]
            pairbatch_line = [pairbatch_line[index] for index in valid_index]
            type_label.extend([valid_relation_type_index[index] for index in valid_index])
            for pair_line in pairbatch_line:
                if not torch.equal(pair_line, padding):
                    if outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) == 1:
                        sentence_concept_tensor.append(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :])
                    elif outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) != 1:
                        sentence_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :], dim=0, keepdim=True))
                    if outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) == 1:
                        answer_concept_tensor.append(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :])
                    elif outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) != 1:
                        answer_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :], dim=0, keepdim=True))
        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sentence_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(answer_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)

        plausible_loss_fct = nn.CrossEntropyLoss()
        type_loss_fct = nn.CrossEntropyLoss()

        type_label = torch.stack(type_label, dim=0)
        # type_label_mask = type_label != 0
        # type_label = type_label[type_label_mask]

        type_logit = self.relation_type_layer(pair_data_input)

        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_labels), plausible_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type_label), type_label.view(-1))
            total_loss = plausible_loss + type_loss
            return total_loss, self.softmax(plausible_logit), self.softmax(type_logit), type_label
        else:
            return self.softmax(plausible_logit), self.softmax(type_logit), type_label


class ElectraForClassificationOnlyTypeWithDownSample(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_labels = config.num_labels
        self.num_relation_type_label = config.num_relation_type
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        # 레이어 하나
        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_labels)
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, plausible_label=None, pair_ids=None, relation_exist_label=None, relation_type_label=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        padding = torch.tensor([[-1, -1], [-1, -1]]).cuda()
        padding_type = torch.tensor(18).cuda()

        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        sentence_concept_tensor = []
        answer_concept_tensor = []

        downsample_type_label = []
        for batch_idx, (pair_batch_line, type_batch_line) in enumerate(zip(pair_ids, relation_type_label)):
            pairbatch_line = [content for content in pair_batch_line if not torch.equal(content, padding)]

            valid_relation_type_index = torch.stack([content for content in type_batch_line if not torch.equal(content, padding_type)], dim=0)
            valid_index = [index for index in range(valid_relation_type_index.size(dim=0))]
            find_index =  [index for index, value in enumerate(valid_relation_type_index) if not torch.equal(value, torch.tensor(0).cuda())]
            remove_index = list(set(valid_index) - set(find_index))
            shuffle(remove_index)
            downsample = remove_index[:4*len(find_index)] + find_index

            pairbatch_line = [pairbatch_line[index] for index in downsample]
            downsample_type_label.extend([valid_relation_type_index[index] for index in downsample])
            for pair_line in pairbatch_line:
                if not torch.equal(pair_line, padding):
                    if outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) == 1:
                        sentence_concept_tensor.append(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :])
                    elif outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) != 1:
                        sentence_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :], dim=0, keepdim=True))
                    if outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) == 1:
                        answer_concept_tensor.append(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :])
                    elif outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) != 1:
                        answer_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :], dim=0, keepdim=True))
        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sentence_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(answer_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)

        plausible_loss_fct = nn.CrossEntropyLoss()
        type_loss_fct = nn.CrossEntropyLoss()

        downsample_type_label = torch.stack(downsample_type_label, dim=0)
        # type_label_mask = type_label != 0
        # type_label = type_label[type_label_mask]

        type_logit = self.relation_type_layer(pair_data_input)

        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_labels), plausible_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type_label), downsample_type_label.view(-1))
            total_loss = plausible_loss + type_loss
            return total_loss, self.softmax(plausible_logit), self.softmax(type_logit), downsample_type_label
        else:
            return self.softmax(plausible_logit), self.softmax(type_logit), downsample_type_label


# 관계 유형만 no relation 제외하고 학습
class ElectraForClassificationOnlyTypeWithDownSampleNoRelation(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_labels = config.num_labels
        self.num_relation_type_label = config.num_relation_type
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        # 레이어 하나
        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_labels)
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, plausible_label=None, pair_ids=None, relation_exist_label=None, relation_type_label=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        padding = torch.tensor([[-1, -1], [-1, -1]]).cuda()
        padding_type = torch.tensor(18).cuda()

        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        sentence_concept_tensor = []
        answer_concept_tensor = []

        downsample_type_label = []
        for batch_idx, (pair_batch_line, type_batch_line) in enumerate(zip(pair_ids, relation_type_label)):
            pairbatch_line = [content for content in pair_batch_line if not torch.equal(content, padding)]

            valid_relation_type_index = torch.stack([content for content in type_batch_line if not torch.equal(content, padding_type)], dim=0)
            valid_index = [index for index in range(valid_relation_type_index.size(dim=0))]
            find_index =  [index for index, value in enumerate(valid_relation_type_index) if not torch.equal(value, torch.tensor(0).cuda())]
            remove_index = list(set(valid_index) - set(find_index))
            shuffle(remove_index)
            downsample = remove_index[:4*len(find_index)] + find_index

            pairbatch_line = [pairbatch_line[index] for index in downsample]
            downsample_type_label.extend([valid_relation_type_index[index] for index in downsample])
            for pair_line in pairbatch_line:
                if not torch.equal(pair_line, padding):
                    if outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) == 1:
                        sentence_concept_tensor.append(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :])
                    elif outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) != 1:
                        sentence_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :], dim=0, keepdim=True))
                    if outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) == 1:
                        answer_concept_tensor.append(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :])
                    elif outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) != 1:
                        answer_concept_tensor.append(
                            torch.sum(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :], dim=0, keepdim=True))
        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sentence_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(answer_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)

        plausible_loss_fct = nn.CrossEntropyLoss()
        type_loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        downsample_type_label = torch.stack(downsample_type_label, dim=0)
        downsample_type_label_mask = downsample_type_label != 0
        downsample_type_label = downsample_type_label[downsample_type_label_mask]
        # type_label_mask = type_label != 0
        # type_label = type_label[type_label_mask]

        type_logit = self.relation_type_layer(pair_data_input)
        type_logit = type_logit[downsample_type_label_mask]

        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_labels), plausible_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type_label), downsample_type_label.view(-1))
            total_loss = plausible_loss + type_loss
            return total_loss, self.softmax(plausible_logit), self.softmax(type_logit), downsample_type_label
        else:
            return self.softmax(plausible_logit), self.softmax(type_logit), downsample_type_label

# 관계 존재 + 관계 유형 다운샘플링 하기 전
class ElectraForClassificationExistAndType(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_labels = config.num_labels             # plausible, neutral, implausible
        self.num_relation_exist_label = config.num_relation_exist     # exist, non exist,
        self.num_relation_type_label = config.num_relation_type   # 17개 관계 / +1로 no relation,
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_labels)
        )
        self.relation_exist_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist_label),
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label),
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_ids=None, attention_mask=None, pair_ids=None, relation_exist_label=None, relation_type_label=None, plausible_label=None):
        # (batch_size, max_length, hidden_size)
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        padding = torch.tensor([[-1, -1], [-1, -1]]).cuda()
        padding_exist = torch.tensor(2).cuda()
        padding_type = torch.tensor(18).cuda()

        relation_existing = torch.tensor(1).cuda()
        # (batch_size, max_document_length, hidden_size) -> (batch_size, num_labels)
        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        sentence_concept_tensor = []
        answer_concept_tensor = []

        exist_label = []
        type_label = []
        for batch_idx, (pair_batch_line, exist_batch_line, type_batch_line) in enumerate(zip(pair_ids, relation_exist_label, relation_type_label)):
            # batch_line : (pair_max_length, 2, 2)
            pairbatch_line = [content for content in pair_batch_line if not torch.equal(content, padding)]
            valid_relation_exist_index = torch.stack([content for content in exist_batch_line if not torch.equal(content, padding_exist)], dim=0)
            valid_relation_type_index = torch.stack([content for content in type_batch_line if not torch.equal(content, padding_type)], dim=0)

            valid_index = [index for index in range(valid_relation_exist_index.size(dim=0))]

            pairbatch_line = [pairbatch_line[index] for index in valid_index]
            exist_label.extend([valid_relation_exist_index[index] for index in valid_index])
            type_label.extend([valid_relation_type_index[index] for index in valid_index])

            for pair_line in pairbatch_line:
                if not torch.equal(pair_line, padding):
                    if outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) == 1:
                        sentence_concept_tensor.append(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :])
                    elif outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) != 1:
                        sentence_concept_tensor.append(torch.sum(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :], dim=0, keepdim=True))
                    if outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) == 1:
                        answer_concept_tensor.append(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :])
                    elif outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) != 1:
                        answer_concept_tensor.append(torch.sum(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :], dim=0, keepdim=True))

        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sentence_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(answer_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)

        plausible_loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.CrossEntropyLoss()
        type_loss_fct = nn.CrossEntropyLoss()

        exist_label = torch.stack(exist_label, dim=0)
        type_label = torch.stack(type_label, dim=0)
        #downsample_type_label_mask = downsample_type_label != 0
        #downsample_type_label = downsample_type_label[downsample_type_label_mask]


        exist_logit = self.relation_exist_layer(pair_data_input)
        type_logit = self.relation_type_layer(pair_data_input)
        #type_logit = type_logit[downsample_type_label_mask]

        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_labels), plausible_label.view(-1))
            exist_loss = exist_loss_fct(exist_logit.view(-1, self.num_relation_exist_label), exist_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type_label), type_label.view(-1))
            total_loss = plausible_loss + exist_loss + type_loss
            # total loss 수정도 생각
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), exist_label, type_label
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), exist_label, type_label


# 둘다 하는데 다운샘플링 포함
class ElectraForClassificationExistAndTypeWithDownSample(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_labels = config.num_labels             # plausible, neutral, implausible
        self.num_relation_exist_label = config.num_relation_exist     # exist, non exist,
        self.num_relation_type_label = config.num_relation_type   # 17개 관계 / +1로 no relation,
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_labels)
        )
        self.relation_exist_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist_label),
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label),
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_ids=None, attention_mask=None, pair_ids=None, relation_exist_label=None, relation_type_label=None, plausible_label=None):
        # (batch_size, max_length, hidden_size)
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        padding = torch.tensor([[-1, -1], [-1, -1]]).cuda()
        padding_exist = torch.tensor(2).cuda()
        padding_type = torch.tensor(18).cuda()

        relation_existing = torch.tensor(1).cuda()
        # (batch_size, max_document_length, hidden_size) -> (batch_size, num_labels)
        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        sentence_concept_tensor = []
        answer_concept_tensor = []

        downsample_exist_label = []
        downsample_type_label = []
        for batch_idx, (pair_batch_line, exist_batch_line, type_batch_line) in enumerate(zip(pair_ids, relation_exist_label, relation_type_label)):
            # batch_line : (pair_max_length, 2, 2)
            pairbatch_line = [content for content in pair_batch_line if not torch.equal(content, padding)]
            valid_relation_exist_index = torch.stack([content for content in exist_batch_line if not torch.equal(content, padding_exist)], dim=0)
            valid_relation_type_index = torch.stack([content for content in type_batch_line if not torch.equal(content, padding_type)], dim=0)

            valid_index = [index for index in range(valid_relation_exist_index.size(dim=0))]
            find_index = [index for index, value in enumerate(valid_relation_exist_index) if not torch.equal(value, torch.tensor(0).cuda())]
            remove_index = list(set(valid_index)-set(find_index))
            shuffle(remove_index)
            downsample = remove_index[:4*len(find_index)]+find_index

            pairbatch_line = [pairbatch_line[index] for index in downsample]
            downsample_exist_label.extend([valid_relation_exist_index[index] for index in downsample])
            downsample_type_label.extend([valid_relation_type_index[index] for index in downsample])

            for pair_line in pairbatch_line:
                if not torch.equal(pair_line, padding):
                    if outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) == 1:
                        sentence_concept_tensor.append(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :])
                    elif outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) != 1:
                        sentence_concept_tensor.append(torch.sum(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :], dim=0, keepdim=True))
                    if outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) == 1:
                        answer_concept_tensor.append(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :])
                    elif outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) != 1:
                        answer_concept_tensor.append(torch.sum(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :], dim=0, keepdim=True))

        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sentence_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(answer_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)

        plausible_loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.CrossEntropyLoss()
        type_loss_fct = nn.CrossEntropyLoss()

        # exist_label = torch.stack(exist_label, dim=0)
        # type_label = torch.stack(type_label, dim=0)
        downsample_exist_label = torch.stack(downsample_exist_label, dim=0)
        downsample_type_label = torch.stack(downsample_type_label, dim=0)
        # downsample_type_label_mask = downsample_type_label != 0
        # downsample_type_label = downsample_type_label[downsample_type_label_mask]


        exist_logit = self.relation_exist_layer(pair_data_input)
        type_logit = self.relation_type_layer(pair_data_input)
        #type_logit = type_logit[downsample_type_label_mask]

        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_labels), plausible_label.view(-1))
            exist_loss = exist_loss_fct(exist_logit.view(-1, self.num_relation_exist_label), downsample_exist_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type_label), downsample_type_label.view(-1))
            total_loss = plausible_loss + exist_loss + type_loss
            # total loss 수정도 생각
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label


# 둘다 하는데 다운샘플링 포함
class ElectraForClassificationExistAndTypeWithDownSampleNoRelationRemove(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_labels = config.num_labels             # plausible, neutral, implausible
        self.num_relation_exist_label = config.num_relation_exist     # exist, non exist,
        self.num_relation_type_label = config.num_relation_type   # 17개 관계 / +1로 no relation,
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)

        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_labels)
        )
        self.relation_exist_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist_label),
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label),
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_ids=None, attention_mask=None, pair_ids=None, relation_exist_label=None, relation_type_label=None, plausible_label=None):
        # (batch_size, max_length, hidden_size)
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        padding = torch.tensor([[-1, -1], [-1, -1]]).cuda()
        padding_exist = torch.tensor(2).cuda()
        padding_type = torch.tensor(18).cuda()

        relation_existing = torch.tensor(1).cuda()
        # (batch_size, max_document_length, hidden_size) -> (batch_size, num_labels)
        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        sentence_concept_tensor = []
        answer_concept_tensor = []

        downsample_exist_label = []
        downsample_type_label = []
        for batch_idx, (pair_batch_line, exist_batch_line, type_batch_line) in enumerate(zip(pair_ids, relation_exist_label, relation_type_label)):
            # batch_line : (pair_max_length, 2, 2)
            pairbatch_line = [content for content in pair_batch_line if not torch.equal(content, padding)]
            valid_relation_exist_index = torch.stack([content for content in exist_batch_line if not torch.equal(content, padding_exist)], dim=0)
            valid_relation_type_index = torch.stack([content for content in type_batch_line if not torch.equal(content, padding_type)], dim=0)

            valid_index = [index for index in range(valid_relation_exist_index.size(dim=0))]
            find_index = [index for index, value in enumerate(valid_relation_exist_index) if not torch.equal(value, torch.tensor(0).cuda())]
            remove_index = list(set(valid_index)-set(find_index))
            shuffle(remove_index)
            downsample = remove_index[:4*len(find_index)]+find_index

            pairbatch_line = [pairbatch_line[index] for index in downsample]
            downsample_exist_label.extend([valid_relation_exist_index[index] for index in downsample])
            downsample_type_label.extend([valid_relation_type_index[index] for index in downsample])

            for pair_line in pairbatch_line:
                if not torch.equal(pair_line, padding):
                    if outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) == 1:
                        sentence_concept_tensor.append(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :])
                    elif outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :].size(dim=0) != 1:
                        sentence_concept_tensor.append(torch.sum(outputs[batch_idx, pair_line[0][0]:pair_line[0][1], :], dim=0, keepdim=True))
                    if outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) == 1:
                        answer_concept_tensor.append(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :])
                    elif outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :].size(dim=0) != 1:
                        answer_concept_tensor.append(torch.sum(outputs[batch_idx, pair_line[1][0]:pair_line[1][1], :], dim=0, keepdim=True))

        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sentence_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(answer_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)

        plausible_loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.CrossEntropyLoss()
        type_loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        # exist_label = torch.stack(exist_label, dim=0)
        # type_label = torch.stack(type_label, dim=0)
        downsample_exist_label = torch.stack(downsample_exist_label, dim=0)
        downsample_type_label = torch.stack(downsample_type_label, dim=0)
        downsample_type_label_mask = downsample_type_label != 0
        downsample_type_label = downsample_type_label[downsample_type_label_mask]


        exist_logit = self.relation_exist_layer(pair_data_input)
        type_logit = self.relation_type_layer(pair_data_input)
        type_logit = type_logit[downsample_type_label_mask]

        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_labels), plausible_label.view(-1))
            exist_loss = exist_loss_fct(exist_logit.view(-1, self.num_relation_exist_label), downsample_exist_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type_label), downsample_type_label.view(-1))
            total_loss = plausible_loss + exist_loss + type_loss
            # total loss 수정도 생각
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label