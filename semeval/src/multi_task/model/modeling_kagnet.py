import dgl.function as fn
import networkx as nx
import torch.utils.data as data
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
from transformers import AutoModel
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer)

GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']
MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


def get_gpt_token_num():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    return len(tokenizer)


def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.
    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m

class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)
        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
        else:
            mask = mask_or_lengths
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = masked_inputs.max(1)[0]
        return max_pooled


class MeanPoolLayer(nn.Module):
    """
    A layer that performs mean pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)
        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
            lengths = mask_or_lengths.float()
        else:
            mask, lengths = mask_or_lengths, (1 - mask_or_lengths.float()).sum(1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), 0.0)
        mean_pooled = masked_inputs.sum(1) / lengths.unsqueeze(-1)
        return mean_pooled


class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.
    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class LSTMTextEncoder(nn.Module):
    pool_layer_classes = {'mean': MeanPoolLayer, 'max': MaxPoolLayer}

    def __init__(self, vocab_size=1, emb_size=300, hidden_size=300, output_size=300, num_layers=2, bidirectional=True,
                 emb_p=0.0, input_p=0.0, hidden_p=0.0, pretrained_emb_or_path=None, freeze_emb=True,
                 pool_function='max', output_hidden_states=False):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.output_hidden_states = output_hidden_states
        assert not bidirectional or hidden_size % 2 == 0

        if pretrained_emb_or_path is not None:
            if isinstance(pretrained_emb_or_path, str):  # load pretrained embedding from a .npy file
                pretrained_emb_or_path = torch.tensor(np.load(pretrained_emb_or_path), dtype=torch.float)
            emb = nn.Embedding.from_pretrained(pretrained_emb_or_path, freeze=freeze_emb)
            emb_size = emb.weight.size(1)
        else:
            emb = nn.Embedding(vocab_size, emb_size)
        self.emb = EmbeddingDropout(emb, emb_p)
        self.rnns = nn.ModuleList([nn.LSTM(emb_size if l == 0 else hidden_size,
                                           (hidden_size if l != num_layers else output_size) // (2 if bidirectional else 1),
                                           1, bidirectional=bidirectional, batch_first=True) for l in range(num_layers)])
        self.pooler = self.pool_layer_classes[pool_function]()

        self.input_dropout = nn.Dropout(input_p)
        self.hidden_dropout = nn.ModuleList([RNNDropout(hidden_p) for _ in range(num_layers)])

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)
        returns: tensor of shape (batch_size, hidden_size)
        """
        assert (lengths > 0).all()
        batch_size, seq_len = inputs.size()
        hidden_states = self.input_dropout(self.emb(inputs))
        all_hidden_states = [hidden_states]
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dropout)):
            # hidden_states = pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
            hidden_states, _ = rnn(hidden_states)
            # hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, total_length=seq_len)
            all_hidden_states.append(hidden_states)
            if l != self.num_layers - 1:
                hidden_states = hid_dp(hidden_states)
        pooled = self.pooler(all_hidden_states[-1], lengths)
        assert len(all_hidden_states) == self.num_layers + 1
        outputs = (all_hidden_states[-1], pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs


class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states
        assert not self.output_token_states or self.model_type in ('bert', 'roberta',)

        if self.model_type in ('lstm',):
            self.module = LSTMTextEncoder(**kwargs, output_hidden_states=True)
            self.sent_dim = self.module.output_size
        else:
            self.module = AutoModel.from_pretrained(model_name, output_hidden_states=True)
            if from_checkpoint is not None:
                self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
            if self.model_type in ('gpt',):
                self.module.resize_token_embeddings(get_gpt_token_num())
            self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        '''
        layer_id: only works for non-LSTM encoders
        output_token_states: if True, return hidden states of specific layer and attention masks
        '''

        if self.model_type in ('lstm',):  # lstm
            input_ids, lengths = inputs
            outputs = self.module(input_ids, lengths)
        elif self.model_type in ('gpt',):  # gpt
            input_ids, cls_token_ids, lm_labels = inputs  # lm_labels is not used
            outputs = self.module(input_ids)
        else:  # bert / xlnet / roberta
            input_ids, attention_mask, token_type_ids, output_mask = inputs
            outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        if self.model_type in ('lstm',):
            sent_vecs = outputs[1]
        elif self.model_type in ('gpt',):
            cls_token_ids = cls_token_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            sent_vecs = hidden_states.gather(1, cls_token_ids).squeeze(1)
        elif self.model_type in ('xlnet',):
            sent_vecs = hidden_states[:, -1]
        else:  # bert / roberta
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = self.module.pooler(hidden_states)
        return sent_vecs, all_hidden_states


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {"h":h}


class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GraphConvLayer, self).__init__()
        self.apply_mode = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mode)
        return g.ndata.pop('h')


class GCNEncoder(nn.Module):
    def __init__(self, concept_dim, hidden_dim, output_dim, pretrained_concept_emd, concept_emd=None):
        super(GCNEncoder, self).__init__()

        self.gcn1 = GraphConvLayer(concept_dim, hidden_dim, F.relu)
        self.gcn2 = GraphConvLayer(hidden_dim, output_dim, F.relu)

        if pretrained_concept_emd is not None and concept_emd is None:
            self.concept_emd = nn.Embedding(pretrained_concept_emd.size(0), pretrained_concept_emd.size(1))
            self.concept_emd.weight.data.copy_(pretrained_concept_emd)
        elif pretrained_concept_emd is None and concept_emd is not None:
            self.concept_emd = concept_emd
        else:
            raise ValueError('invalid pretrained_concept_emd/concept_emd')

    def forward(self, g):
        features = self.concept_emd(g.ndata["cncpt_ids"])
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        g.ndata['h'] = x
        return g


class GCNLayer(nn.Module):

    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        assert input_size == output_size

        self.w = nn.Parameter(torch.zeros(input_size, output_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.normal_(self.w_vs, mean=0, std=np.sqrt(2.0 / (input_size + output_size)))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, normalized_adj_t):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        normalized_adj_t: tensor of shape (b_sz, n_node, n_node)
            normalized_adj_t[:, j, i] ==  1/n indicates a directed edge i --> j and in_degree(j) == n
        """

        bs, n_node, _ = inputs.size()

        output = inputs.matmul(self.w)  # (b_sz, n_node, o_size)
        output = normalized_adj_t.bmm(output)
        output = self.activation(output)
        output = self.dropout(output)
        return output


class GCN(nn.Module):

    def __init__(self, input_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(input_size, input_size, dropout) for l in range(num_layers + 1)])

    def forward(self, inputs, adj):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        adj: tensor of shape (b_sz, n_head, n_node, n_node)
            we assume the identity matrix representing self loops are already added to adj
        """
        bs, n_node, _ = adj.size()

        in_degree = torch.max(adj.sum(1), adj.new_ones(()))
        adj_t = adj.transpose(1, 2)
        normalized_adj_t = (adj_t / in_degree.unsqueeze(-1))  # (bz, n_node, n_node)
        assert ((torch.abs(normalized_adj_t.sum(2) - 1) < 1e-5) | (torch.abs(normalized_adj_t.sum(2)) < 1e-5)).all()

        output = inputs
        for layer in self.layers:
            output = layer(output, normalized_adj_t)
        return output


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class KnowledgeAwareGraphNetwork(nn.Module):
    def __init__(self, sent_dim, concept_dim, relation_dim, concept_num, relation_num,
                 qas_encoded_dim, pretrained_concept_emd, pretrained_relation_emd,
                 lstm_dim, lstm_layer_num, graph_hidden_dim, graph_output_dim,
                 dropout=0.1, bidirect=True, num_random_paths=None, path_attention=True,
                 qa_attention=True):
        super(KnowledgeAwareGraphNetwork, self).__init__()
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim
        self.sent_dim = sent_dim
        self.concept_num = concept_num
        self.relation_num = relation_num
        self.qas_encoded_dim = qas_encoded_dim
        self.pretrained_concept_emd = pretrained_concept_emd
        self.pretrained_relation_emd = pretrained_relation_emd
        self.lstm_dim = lstm_dim
        self.lstm_layer_num = lstm_layer_num
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_output_dim = graph_output_dim
        self.dropout = dropout
        self.bidirect = bidirect
        self.num_random_paths = num_random_paths
        self.path_attention = path_attention
        self.qa_attention = qa_attention

        self.concept_emd = nn.Embedding(concept_num, concept_dim)
        self.relation_emd = nn.Embedding(relation_num, relation_dim)

        if pretrained_concept_emd is not None:
            self.concept_emd.weight.data.copy_(pretrained_concept_emd)
        else:
            bias = np.sqrt(6.0 / self.concept_dim)
            nn.init.uniform_(self.concept_emd.weight, -bias, bias)

        if pretrained_relation_emd is not None:
            self.relation_emd.weight.data.copy_(pretrained_relation_emd)
        else:
            bias = np.sqrt(6.0 / self.relation_dim)
            nn.init.uniform_(self.relation_emd.weight, -bias, bias)

        self.lstm = nn.LSTM(input_size=graph_output_dim + concept_dim + relation_dim,
                            hidden_size=lstm_dim,
                            num_layers=lstm_layer_num,
                            bidirectional=bidirect,
                            dropout=dropout,
                            batch_first=True)

        if bidirect:
            self.lstm_dim = lstm_dim * 2

        self.qas_encoder = nn.Sequential(
            nn.Linear(2 * (concept_dim + graph_output_dim) + sent_dim, self.qas_encoded_dim * 2),  # binary classification
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(self.qas_encoded_dim * 2, self.qas_encoded_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        if self.path_attention:  # TODO: can be optimized by using nn.BiLinaer
            self.qas_pathlstm_att = nn.Linear(self.qas_encoded_dim, self.lstm_dim)  # transform qas vector to query vectors
            self.qas_pathlstm_att.apply(weight_init)

        if self.qa_attention:
            self.sent_ltrel_att = nn.Linear(sent_dim, self.qas_encoded_dim)  # transform sentence vector to query vectors
            self.sent_ltrel_att.apply(weight_init)

        self.hidden2output = nn.Sequential(
            nn.Linear(self.qas_encoded_dim + self.lstm_dim + self.sent_dim, 1),  # binary classification
        )

        self.lstm.apply(weight_init)
        self.qas_encoder.apply(weight_init)
        self.hidden2output.apply(weight_init)

        self.graph_encoder = GCNEncoder(self.concept_dim, self.graph_hidden_dim, self.graph_output_dim,
                                        pretrained_concept_emd=None, concept_emd=self.concept_emd)

    def forward(self, s_vec_batched, qa_pairs_batched, cpt_paths_batched, rel_paths_batched, qa_path_num_batched, path_len_batched, graphs, concept_mapping_dicts, ana_mode=False):
        output_graphs = self.graph_encoder(graphs)
        new_concept_embed = torch.cat((output_graphs.ndata["h"], s_vec_batched.new_zeros((1, self.graph_output_dim))))  # len(output_concept_embeds) as padding

        final_vecs = []

        if ana_mode:
            path_att_scores = []
            qa_pair_att_scores = []

        n_qa_pairs = [len(t) for t in qa_pairs_batched]
        total_qa_pairs = sum(n_qa_pairs)
        s_vec_expanded = s_vec_batched.new_zeros((total_qa_pairs, s_vec_batched.size(1)))
        i = 0
        for n, s_vec in zip(n_qa_pairs, s_vec_batched):
            j = i + n
            s_vec_expanded[i:j] = s_vec
            i = j
        qa_ids_batched = torch.cat(qa_pairs_batched, 0)  # N x 2
        qa_vecs = self.concept_emd(qa_ids_batched).view(total_qa_pairs, -1)
        new_qa_ids = []
        for qa_ids, mdict in zip(qa_pairs_batched, concept_mapping_dicts):
            id_mapping = lambda x: mdict.get(x, len(new_concept_embed) - 1)
            new_qa_ids += [[id_mapping(q), id_mapping(a)] for q, a in qa_ids]
        new_qa_ids = torch.tensor(new_qa_ids, device=s_vec_batched.device)
        new_qa_vecs = new_concept_embed[new_qa_ids].view(total_qa_pairs, -1)
        raw_qas_vecs = torch.cat((qa_vecs, new_qa_vecs, s_vec_expanded), dim=1)  # all the qas triple vectors associated with a statement
        qas_vecs_batched = self.qas_encoder(raw_qas_vecs)
        if self.path_attention:
            query_vecs_batched = self.qas_pathlstm_att(qas_vecs_batched)
        flat_cpt_paths_batched = torch.cat(cpt_paths_batched, 0)
        mdicted_cpaths = []
        for cpt_path in flat_cpt_paths_batched:
            mdicted_cpaths.append([id_mapping(c) for c in cpt_path])
        mdicted_cpaths = torch.tensor(mdicted_cpaths, device=s_vec_batched.device)

        new_batched_all_qa_cpt_paths_embeds = new_concept_embed[mdicted_cpaths]
        batched_all_qa_cpt_paths_embeds = self.concept_emd(torch.cat(cpt_paths_batched, 0))  # old concept embed

        batched_all_qa_cpt_paths_embeds = torch.cat((batched_all_qa_cpt_paths_embeds, new_batched_all_qa_cpt_paths_embeds), 2)

        batched_all_qa_rel_paths_embeds = self.relation_emd(torch.cat(rel_paths_batched, 0))  # N_PATHS x D x MAX_PATH_LEN

        batched_all_qa_cpt_rel_path_embeds = torch.cat((batched_all_qa_cpt_paths_embeds,
                                                        batched_all_qa_rel_paths_embeds), 2)

        # if False then abiliate the LSTM
        if True:
            batched_lstm_outs, _ = self.lstm(batched_all_qa_cpt_rel_path_embeds)

        else:
            batched_lstm_outs = s_vec.new_zeros((batched_all_qa_cpt_rel_path_embeds.size(0),
                                                 batched_all_qa_cpt_rel_path_embeds.size(1),
                                                 self.lstm_dim))
        b_idx = torch.arange(batched_lstm_outs.size(0)).to(batched_lstm_outs.device)
        batched_lstm_outs = batched_lstm_outs[b_idx, torch.cat(path_len_batched, 0) - 1, :]

        qa_pair_cur_start = 0
        path_cur_start = 0
        # for each question-answer statement
        for s_vec, qa_ids, cpt_paths, rel_paths, mdict, qa_path_num, path_len in zip(s_vec_batched, qa_pairs_batched, cpt_paths_batched,
                                                                                     rel_paths_batched, concept_mapping_dicts, qa_path_num_batched,
                                                                                     path_len_batched):  # len = batch_size * num_choices

            n_qa_pairs = qa_ids.size(0)
            qa_pair_cur_end = qa_pair_cur_start + n_qa_pairs

            if n_qa_pairs == 0 or False:  # if "or True" then we can do ablation study
                raw_qas_vecs = torch.cat([s_vec.new_zeros((self.concept_dim + self.graph_output_dim) * 2), s_vec], 0).view(1, -1)
                qas_vecs = self.qas_encoder(raw_qas_vecs)
                latent_rel_vecs = torch.cat((qas_vecs, s_vec.new_zeros(1, self.lstm_dim)), dim=1)
            else:
                pooled_path_vecs = []
                qas_vecs = qas_vecs_batched[qa_pair_cur_start:qa_pair_cur_end]
                for j in range(n_qa_pairs):
                    if self.path_attention:
                        query_vec = query_vecs_batched[qa_pair_cur_start + j]

                    path_cur_end = path_cur_start + qa_path_num[j]

                    # pooling over all paths for a certain (question concept, answer concept) pair
                    blo = batched_lstm_outs[path_cur_start:path_cur_end]
                    if self.path_attention:  # TODO: use an attention module for better readability
                        att_scores = torch.mv(blo, query_vec)  # path-level attention scores
                        norm_att_scores = F.softmax(att_scores, 0)
                        att_pooled_path_vec = torch.mv(blo.t(), norm_att_scores)
                        if ana_mode:
                            path_att_scores.append(norm_att_scores)
                    else:
                        att_pooled_path_vec = blo.mean(0)

                    path_cur_start = path_cur_end
                    pooled_path_vecs.append(att_pooled_path_vec)

                pooled_path_vecs = torch.stack(pooled_path_vecs, 0)
                latent_rel_vecs = torch.cat((qas_vecs, pooled_path_vecs), 1)  # qas and KE-qas

            # pooling over all (question concept, answer concept) pairs
            if self.path_attention:
                sent_as_query = self.sent_ltrel_att(s_vec)  # sent attend on qas
                r_att_scores = torch.mv(qas_vecs, sent_as_query)  # qa-pair-level attention scores
                norm_r_att_scores = F.softmax(r_att_scores, 0)
                if ana_mode:
                    qa_pair_att_scores.append(norm_r_att_scores)
                final_vec = torch.mv(latent_rel_vecs.t(), norm_r_att_scores)
            else:
                final_vec = latent_rel_vecs.mean(0).to(s_vec.device)  # mean pooling
            final_vecs.append(torch.cat((final_vec, s_vec), 0))

            qa_pair_cur_start = qa_pair_cur_end

        logits = self.hidden2output(torch.stack(final_vecs))
        if not ana_mode:
            return logits
        else:
            return logits, (path_att_scores, qa_pair_att_scores)


class LMKagNet(nn.Module):
    # qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts
    def __init__(self, model_name, concept_dim, relation_dim, concept_num, relation_num,
                 qas_encoded_dim, pretrained_concept_emb, pretrained_relation_emb,
                 lstm_dim, lstm_layer_num, graph_hidden_dim, graph_output_dim,
                 dropout=0.1, bidirect=True, num_random_paths=None, path_attention=True,
                 qa_attention=True, encoder_config={}):
        super().__init__()

        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = KnowledgeAwareGraphNetwork(self.encoder.sent_dim, concept_dim, relation_dim, concept_num, relation_num,
                                                  qas_encoded_dim, pretrained_concept_emb, pretrained_relation_emb,
                                                  lstm_dim, lstm_layer_num, graph_hidden_dim, graph_output_dim,
                                                  dropout=dropout, bidirect=bidirect, num_random_paths=num_random_paths, path_attention=path_attention,
                                                  qa_attention=qa_attention)

    def forward(self, *inputs, layer_id=-1):
        """
        sent_vecs: (batch_size, num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)
        adj: (batch_size, num_choice, n_head, n_node, n_node)
        adj_lengths: (batch_size, num_choice)
        node_type_ids: (batch_size, num_choice n_node)
        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-7]] + inputs[-7:]  # merge the batch dimension and the num_choice dimension
        print(len(inputs))
        *lm_inputs, qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts = inputs
        print([len(x[0]) for x in [qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts]])
        print([x.device for x in [qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts]])
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits, attn = self.decoder(sent_vecs.view(bs, nc, -1).to(qa_pair_data.device), qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts)
        logits = logits.view(bs, nc)
        return logits, attn
