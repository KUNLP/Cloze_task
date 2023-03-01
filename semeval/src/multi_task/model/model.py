from typing import List

from transformers import RobertaModel, RobertaPreTrainedModel
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
import pandas as pd
from random import shuffle

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class RobertaForMultiTaskLearning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels                     # plausible, neutral, implausible
        self.num_relation_exist = config.num_relation_exist         # exist, non exist
        self.num_relation_type = config.num_relation_type           # conceptnet 기준으로 17개
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)

        self.plausible_layer = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=self.hidden_size, out_features=self.num_labels)
            )
        )
        self.relation_exist_layer = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist)
            )
        )
        self.relation_type_layer = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type)
            )
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state

        #cls token
        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        #페어들간에 봐야함
        if labels is not None:
            plausible_loss_func = nn.CrossEntropyLoss()
            plausible_loss = plausible_loss_func(plausible_logit, labels)
            return plausible_loss, self.softmax(plausible_logit)
        else:
            return self.softmax(plausible_logit)

# 한번에 로스 구하는거 말고 하나씩 돌면서 평균내기
class ElectraForMultiTaskClassification(ElectraPreTrainedModel):
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
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist_label),
            nn.Dropout(0.1)
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label),
            nn.Dropout(0.1)
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
            total_loss = plausible_loss + 0.5*exist_loss + 0.5*type_loss
            # total loss 수정도 생각
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label

# 아래거는 exist랑 type개수가 같음
class ElectraForMultiTaskClassification4(ElectraPreTrainedModel):
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
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist_label),
            nn.Dropout(0.1)
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label),
            nn.Dropout(0.1)
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

        exist_logit = self.relation_exist_layer(pair_data_input)
        type_logit = self.relation_type_layer(pair_data_input)

        plausible_loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.CrossEntropyLoss()
        type_loss_fct = nn.CrossEntropyLoss()

        downsample_exist_label = torch.stack(downsample_exist_label, dim=0)
        downsample_type_label = torch.stack(downsample_type_label, dim=0)
        if plausible_label is not None:
            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_labels), plausible_label.view(-1))
            exist_loss = exist_loss_fct(exist_logit.view(-1, self.num_relation_exist_label), downsample_exist_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type_label), downsample_type_label.view(-1))
            total_loss = plausible_loss + 0.5*exist_loss + 0.5*type_loss
            # total loss 수정도 생각
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label


class ElectraForMultiTaskClassification3(ElectraPreTrainedModel):
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
            nn.Linear(in_features=self.hidden_size, out_features=1),
            nn.Sigmoid()
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, pair_ids=None, relation_exist_label=None, relation_type_label=None, plausible_label=None):
        # (batch_size, max_length, hidden_size)
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[0])

        padding = torch.tensor([[-1, -1], [-1, -1]]).cuda()
        # (batch_size, max_document_length, hidden_size) -> (batch_size, num_labels)
        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        #마스크 씌우기 전에 일단 컨셉 페어부터 플랫하게 피고 해보는건?
        pair_ids = pair_ids.view(-1,2,2)
        row_line = 0
        ids_row = []
        for i in range(pair_ids.size(dim=0)):
            if ((i % 32) == 0) and (i != 0):
                row_line += 1
            if not torch.equal(pair_ids[i, :, :], padding):
                pair_ids[i, :, :] += (32 * row_line)
                ids_row.append(pair_ids[i,:,:] + (32*row_line))

        sentence_concept_start_end = pair_ids[:, 0, :]
        answer_concept_start_end = pair_ids[:, 1, :]

        proceed_outputs = outputs.view(-1, self.hidden_size)
        sent_concept_tensor = []
        ans_concept_tensor = []
        for sent_concept, ans_concept in zip(sentence_concept_start_end, answer_concept_start_end):
            if not (torch.equal(sent_concept, torch.tensor([-1,-1]).cuda()) and torch.equal(ans_concept, torch.tensor([-1, -1]).cuda())):
                if (proceed_outputs[sent_concept[0]:sent_concept[1], :].size(0) != 1):
                    sent_concept_tensor.append(torch.sum(proceed_outputs[sent_concept[0]:sent_concept[1], :], dim=0, keepdim=True))
                else:
                    sent_concept_tensor.append(proceed_outputs[sent_concept[0]:sent_concept[1], :])
                if (proceed_outputs[ans_concept[0]:ans_concept[1], :].size(0) != 1):
                    ans_concept_tensor.append(torch.sum(proceed_outputs[ans_concept[0]:ans_concept[1], :], dim=0, keepdim=True))
                else:
                    ans_concept_tensor.append(proceed_outputs[ans_concept[0]:ans_concept[1], :])


        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sent_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(ans_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)


        exist_logit = self.relation_exist_layer(pair_data_input)
        type_logit = self.relation_type_layer(pair_data_input)

        plausible_loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.BCELoss()
        type_loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        if plausible_label is not None:
            # flat relation exist, type label
            relation_exist_label_mask = relation_exist_label != 2
            relation_type_label_mask = relation_type_label != 18
            relation_exist_label = relation_exist_label[relation_exist_label_mask]
            relation_type_label = relation_type_label[relation_type_label_mask]

            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_plausible_labels), plausible_label.view(-1))
            exist_loss = exist_loss_fct(exist_logit.view(-1), relation_exist_label.view(-1))
            #exist_loss = exist_loss_fct(exist_logit.view(-1, self.num_relation_exist_label), relation_exist_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type_label), relation_type_label.view(-1))

            total_loss = plausible_loss + exist_loss + type_loss
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit)
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit)


class ElectraForMultiTaskClassification2(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels             # plausible, neutral, implausible
        self.num_relation_exist = config.num_relation_exist     # exist, non exist,
        self.num_relation_type = config.num_relation_type   # 17개 관계 / +1로 no relation,
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)
        '''
        self.plausible_layer = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=self.hidden_size, out_features=self.num_labels)
            )
        )
        '''
        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_labels)
        )
        self.relation_exist_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist)
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            #nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type)
        )
        '''
        self.relation_exist_layer = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist)
            )
        )
        '''
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, concept_pair_ids=None, relation_exist=None, relation_type=None, plausible_label=None):
        # (batch_size, max_length, hidden_size)
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[0])

        padding = torch.tensor([[-1, -1], [-1, -1]]).cuda()
        # cls token
        # (batch_size, max_document_length, hidden_size) -> (batch_size, num_labels)
        plausible_logit = self.plausible_layer(outputs[:, 0, :])

        #마스크 씌우기 전에 일단 컨셉 페어부터 플랫하게 피고 해보는건?
        test_pair_ids = concept_pair_ids.view(-1,2,2)
        row_line = 0
        for i in range(test_pair_ids.size(dim=0)):
            if ((i % 32) == 0) and (i != 0):
                row_line += 1
            if not torch.equal(test_pair_ids[i, :, :], padding):
                test_pair_ids[i, :, :] += (32 * row_line)

        sentence_concept_start_end = test_pair_ids[:, 0, :]
        answer_concept_start_end = test_pair_ids[:, 1, :]

        proceed_outputs = outputs.view(-1, self.hidden_size)
        sent_concept_tensor = []
        ans_concept_tensor = []
        for sent_concept, ans_concept in zip(sentence_concept_start_end, answer_concept_start_end):
            if not (torch.equal(sent_concept, torch.tensor([-1,-1]).cuda()) and torch.equal(ans_concept, torch.tensor([-1, -1]).cuda())):
                if (proceed_outputs[sent_concept[0]:sent_concept[1]+1,:].size(0) != 1) and \
                        (sent_concept[0] < sent_concept[1] + 1):
                    sent_concept_tensor.append(torch.mean(proceed_outputs[sent_concept[0]:sent_concept[1] + 1, :], dim=0, keepdim=True))
                elif (proceed_outputs[sent_concept[0]:sent_concept[1]+1,:].size(0) == 1) and \
                        (sent_concept[0] < sent_concept[1] + 1):
                    sent_concept_tensor.append(proceed_outputs[sent_concept[0]:sent_concept[1] + 1, :])

                # 여기서 정답 컨셉 가져올때 ans_concept[0] < ans_concept[1] + 1 이 아닌경우가 있다.
                if (proceed_outputs[ans_concept[0]:ans_concept[1]+1,:].size(0) != 1) and \
                        (ans_concept[0] < ans_concept[1] + 1):
                    ans_concept_tensor.append(torch.mean(proceed_outputs[ans_concept[0]:ans_concept[1]+1,:],dim=0, keepdim=True))
                elif (proceed_outputs[ans_concept[0]:ans_concept[1]+1,:].size(0) == 1) and \
                        (ans_concept[0] < ans_concept[1] + 1):
                    ans_concept_tensor.append(proceed_outputs[ans_concept[0]:ans_concept[1]+1,:])
                else:
                    print("ans_concept idx : {}, {}".format(ans_concept[0], ans_concept[1]+1))
                    print(proceed_outputs[ans_concept[0]:ans_concept[1]+1,:])

        # (num of data, hidden size)
        sent_concept_tensor = torch.cat(sent_concept_tensor, dim=0)
        ans_concept_tensor = torch.cat(ans_concept_tensor, dim=0)
        pair_data_input = torch.cat([sent_concept_tensor, ans_concept_tensor], dim=1)


        exist_logit = self.relation_exist_layer(pair_data_input)
        type_logit = self.relation_type_layer(pair_data_input)

        plausible_loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.CrossEntropyLoss(ignore_index=2)
        type_loss_fct = nn.CrossEntropyLoss(ignore_index=18)


        if plausible_label is not None:
            # flat relation exist, type label
            relation_exist_label_mask = relation_exist != 2
            relation_type_label_mask = relation_type != 18
            relation_exist_label = relation_exist[relation_exist_label_mask]
            relation_type_label = relation_type[relation_type_label_mask]

            plausible_loss = plausible_loss_fct(plausible_logit.view(-1, self.num_labels), plausible_label.view(-1))
            exist_loss = exist_loss_fct(exist_logit.view(-1, self.num_relation_exist), relation_exist_label.view(-1))
            type_loss = type_loss_fct(type_logit.view(-1, self.num_relation_type), relation_type_label.view(-1))

            total_loss = plausible_loss + exist_loss + type_loss
            return total_loss, plausible_logit, exist_logit, type_logit
        else:
            return plausible_logit, exist_logit, type_logit


        '''
        test_proceed_outputs = proceed_outputs[test_data_mask]


        # 일단 이런거 생각하고 있다.
        # 이렇게하면 (batch_size, max_pair_length, 2, 2)인데 True, False 로 패딩 아닌부분 다 가져온다.
        data_mask = concept_pair_ids != padding
        # (batch_size, max_pair_length, 2 , 2) -> (number of pair data,)
        concept_pair_ids_output = concept_pair_ids[data_mask]
        concept_pair_ids_output = concept_pair_ids_output.view(-1, 4)
        # (number of sentence concept, start end position) / (number of answer concept, start end position)
        sentence_pair_ids, answer_pair_ids = concept_pair_ids_output[:, :2], concept_pair_ids_output[:, 2:]
        #sentence_stack_outputs = outputs[sentence_pair_ids]


        relation_exist_label_mask = relation_exist != 2
        relation_type_label_mask = relation_type != 18
        relation_exist_label_flat = relation_exist[relation_exist_label_mask]
        relation_type_label_flat = relation_type[relation_type_label_mask]

        #elation exist logit 리스트 만들고 로짓값들 다 저장해서 리턴 값으로 보내주자. 그리고 로스는 1/n로 최종 로스값
        #concept_pair_ids : (batch_size, 패딩포함 max_length, [start, end], [start, end]) ex) (10, 512, 2, 2)
        batch_sentence_tensor = [] # (batch_size, sentence_concept개수, 1 토큰 하나로 줄여놓음, hidden_size )
        batch_answer_tensor = [] #



        exist_logit_list = []
        type_logit_list = []
        for batch_idx, batch in enumerate(concept_pair_ids):
            #이렇게 하면 batch (512, 2, 2) -> padding 포함 맥스 랭스
            sentence_tensor = []
            answer_tensor = []
            for pair in batch: # shape : tensor([[3, 3], [38, 38]]) #한 문장 안에 있는 페어들 개수
                sentence_start_end, answer_start_end = pair # shape : tensor([38, 38])
                if torch.equal(pair, padding):
                    break
                sen_tensor = outputs[batch_idx, sentence_start_end[0]:sentence_start_end[1]+1, :] # 1,768 or 2~3, 768)
                ans_tensor = outputs[batch_idx, answer_start_end[0]:answer_start_end[1]+1, :]
                if sen_tensor.size(dim=0) != 1:
                    #temp = sen_tensor.mean(dim=0)
                    sen_tensor = sen_tensor.mean(dim=0).unsqueeze(0)
                if ans_tensor.size(dim=0) != 1:
                    ans_tensor = ans_tensor.mean(dim=0).unsqueeze(0)
                sentence_tensor.append(sen_tensor)
                answer_tensor.append(ans_tensor)
            #if sentence_tensor:
            sentence_tensor = torch.stack(sentence_tensor, dim=0)
            #else:
            #    sentence_tensor = outputs[batch_idx, 0, :]
            #   sentence_tensor = torch.stack(sentence_tensor, dim=0)
            #   RuntimeError: stack expects a non-empty TensorList
            #  여기서 에러가 ...
            #if answer_tensor:
            answer_tensor = torch.stack(answer_tensor, dim=0)
            #else:
            #    answer_tensor = outputs[batch_idx, 0, :]

            batch_sentence_tensor.append(sentence_tensor)
            batch_answer_tensor.append(answer_tensor)

            concat_pair = torch.cat((sentence_tensor, answer_tensor), dim=2) # (33, 1, 768*2)

            exist_logit = self.relation_exist_layer(concat_pair)
            type_logit = self.relation_type_layer(concat_pair)
            exist_logit_list.append(exist_logit)
            type_logit_list.append(type_logit)

        loss_fct = nn.CrossEntropyLoss()
        exist_loss_fct = nn.CrossEntropyLoss(ignore_index=2)
        type_loss_fct = nn.CrossEntropyLoss(ignore_index=18)

        exist_losses = []
        type_losses = []
        for idx, (exist_logit, type_logit, exist_label, type_label) in enumerate(zip(exist_logit_list, type_logit_list, relation_exist, relation_type)):
            exist_data = exist_logit.size(dim=0)  # 어디까지가 데이터인가.
            type_data = type_logit.size(dim=0)
            exist_logit_view = exist_logit.view(-1, self.num_relation_exist)
            type_logit_view = type_logit.view(-1, self.num_relation_type)

            exist_loss = exist_loss_fct(self.softmax(exist_logit_view), exist_label[:exist_data].view(-1))
            type_loss = type_loss_fct(self.softmax(type_logit_view), type_label[:type_data].view(-1))
        '''

        '''
            exist_losses.append(exist_loss)
            #exist_losses.append(exist_loss if torch.exist_loss)

            type_losses.append(type_loss)
        '''

        # relation type 도 동일하게
        '''
        # exist와 type은 1/n 으로해서 로스 값
        exist_loss_mean = torch.mean(torch.stack(exist_losses))
        type_loss_mean = torch.mean(torch.stack(type_losses))
        if plausible_label is not None:
            plausible_loss = loss_fct(plausible_logit.view(-1, self.num_labels), plausible_label.view(-1))
            total_loss = plausible_loss + exist_loss_mean + type_loss_mean
            return total_loss, plausible_logit, exist_logit_list, type_logit_list
        else:
            return plausible_logit, exist_logit_list, type_logit_list

        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, position_ids=position_ids, output_hidden_states=False)
        # (batch_size, max_document_length, hidden_size)
        electra_output = outputs[0]

        # sep 토큰에 대응하는 index == 102
        sep_mask = input_ids == 102
        # (batch_size, max_document_length, hidden_size) -> (number_of_sep_tokens, hidden_size)
        sep_output = electra_output[sep_mask]
        sep_output = self.dropout(sep_output)

        # (number_of_sep_tokens, hidden_size) -> (number_of_sep_tokens, num_classes)
        final_output = self.linear(sep_output)

        if labels is not None:
            label_mask = labels != 0
            # (batch_size, max_number_of_sentences) -> (number_of_sep_tokens, )
            labels = labels[label_mask]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(final_output, labels)

            return self.softmax(final_output), loss
        else:
            return self.softmax(final_output)
        '''


class BertForMultiTaskClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_plausible_labels = config.num_labels             # plausible, neutral, implausible
        self.num_relation_exist_label = config.num_relation_exist     # exist, non exist,
        self.num_relation_type_label = config.num_relation_type   # 17개 관계 / +1로 no relation,
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)

        self.plausible_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_plausible_labels)
        )
        self.relation_exist_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_exist_label),
            nn.Dropout(0.1)
        )
        self.relation_type_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_relation_type_label),
            nn.Dropout(0.1)
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
            total_loss = plausible_loss + 0.5*exist_loss + 0.5*type_loss
            # total loss 수정도 생각
            return total_loss, self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label
        else:
            return self.softmax(plausible_logit), self.softmax(exist_logit), self.softmax(type_logit), downsample_exist_label, downsample_type_label