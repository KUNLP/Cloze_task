from src.use_kagnet.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
import torch.nn as nn
import torch
import numpy as np
from src.use_kagnet.data_utils import *

def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = nn.GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': nn.GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn

class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class RGCNLayer(nn.Module):

    def __init__(self, n_head, n_basis, input_size, output_size, dropout=0.1, diag_decompose=False):
        super().__init__()
        self.n_head = n_head
        self.n_basis = n_basis
        self.output_size = output_size
        self.diag_decompose = diag_decompose

        assert input_size == output_size

        if diag_decompose and (input_size != output_size):
            raise ValueError('If diag_decompose=True then input size must equaul to output size')
        if diag_decompose and n_basis:
            raise ValueError('diag_decompose and n_basis > 0 cannot be true at the same time')

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.zeros(input_size, n_head))
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.zeros(input_size, output_size * n_head))
        else:
            self.w_vs = nn.Parameter(torch.zeros(input_size, output_size, n_basis))
            self.w_vs_co = nn.Parameter(torch.zeros(n_basis, n_head))
            nn.init.xavier_uniform_(self.w_vs_co)
        nn.init.normal_(self.w_vs, mean=0, std=np.sqrt(2.0 / (input_size + output_size)))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, normalized_adj_t):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        normalized_adj_t: tensor of shape (b_sz*n_head, n_node, n_node)
            normalized_adj_t[:, j, i] ==  1/n indicates a directed edge i --> j and in_degree(j) == n
        """

        o_size, n_head, n_basis = self.output_size, self.n_head, self.n_basis
        bs, n_node, _ = inputs.size()

        if self.diag_decompose:
            output = (inputs.unsqueeze(-1) * self.w_vs).view(bs, n_node, o_size, n_head)  # b_sz x n_node x n_head x o_size
        elif n_basis == 0:
            w_vs = self.w_vs
            output = inputs.matmul(w_vs).view(bs, n_node, o_size, n_head)  # b_sz x n_node x n_head x o_size
        else:
            w_vs = self.w_vs.matmul(self.w_vs_co).view(-1, o_size * n_head)
            output = inputs.matmul(w_vs).view(bs, n_node, o_size, n_head)  # b_sz x n_node x n_head x o_size

        output = output.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, o_size)  # (b_sz*n_head) x n_node x o_size
        output = normalized_adj_t.bmm(output).view(bs, n_head, n_node, o_size).sum(1)  # b_sz x n_node x dv
        output = self.activation(output)
        output = self.dropout(output)
        return output


class RGCN(nn.Module):

    def __init__(self, input_size, num_heads, num_basis, num_layers, dropout, diag_decompose):
        super().__init__()
        self.layers = nn.ModuleList([RGCNLayer(num_heads, num_basis, input_size, input_size,
                                               dropout, diag_decompose=diag_decompose) for l in range(num_layers + 1)])

    def forward(self, inputs, adj):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        adj: tensor of shape (b_sz, n_head, n_node, n_node)
            we assume the identity matrix representating self loops are already added to adj
        """
        bs, n_head, n_node, _ = adj.size()

        in_degree = torch.max(adj.sum(2), adj.new_ones(()))
        adj_t = adj.transpose(2, 3)
        normalized_adj_t = (adj_t / in_degree.unsqueeze(3)).view(bs * n_head, n_node, n_node)
        assert ((torch.abs(normalized_adj_t.sum(2) - 1) < 1e-5) | (torch.abs(normalized_adj_t.sum(2)) < 1e-5)).all()

        output = inputs
        for layer in self.layers:
            output = layer(output, normalized_adj_t)
        return output


class RGCNNet(nn.Module):

    def __init__(self, num_concepts, num_relations, num_basis, sent_dim, concept_dim, concept_in_dim, freeze_ent_emb,
                 num_gnn_layers, num_attention_heads, fc_dim, num_fc_layers, p_gnn, p_fc,
                 pretrained_concept_emb=None, diag_decompose=False, ablation=None):
        super().__init__()
        self.ablation = ablation

        self.concept_emb = CustomizedEmbedding(concept_num=num_concepts, concept_out_dim=concept_dim, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb, use_contextualized=False)
        gnn_dim = concept_dim
        self.rgcn = RGCN(gnn_dim, num_relations, num_basis, num_gnn_layers, p_gnn, diag_decompose)
        self.pool_layer = MultiheadAttPoolLayer(num_attention_heads, sent_dim, gnn_dim)
        self.fc = MLP(gnn_dim + sent_dim, fc_dim, 3, num_fc_layers, p_fc, True)

    def forward(self, sent_vecs, concepts, adj, adj_lengths):
        """
        sent_vecs: (batch_size, d_sent)
        concepts: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)

        returns: (batch_size, 1)
        """
        bs, n_node = concepts.size()
        gnn_input = self.concept_emb(concepts)
        # node_type_embed = sent_vecs.new_zeros((bs, n_node, self.node_type_emb_dim))
        # gnn_input = torch.cat((gnn_input, node_type_embed), -1)
        gnn_output = self.rgcn(gnn_input, adj)

        adj_lengths = torch.max(adj_lengths, adj_lengths.new_ones(()))  # a temporary solution to avoid zero node
        mask = torch.arange(concepts.size(1), device=adj.device).unsqueeze(0) >= adj_lengths.unsqueeze(1)
        # pooled size (1, 100)
        # sent_vecs (1, 1024)
        pooled, pool_attn = self.pool_layer(sent_vecs, gnn_output, mask)
        # pooled = sent_vecs.new_zeros((sent_vecs.size(0), self.hid2out.weight.size(1) - sent_vecs.size(1)))
        logits = self.fc(torch.cat((pooled, sent_vecs), 1))
        return logits, pool_attn


class LMRGCN(nn.Module):
    def __init__(self, model_name, num_concepts, num_relations, num_basis, concept_dim, concept_in_dim, freeze_ent_emb,
                 num_gnn_layers, num_attention_heads, fc_dim, num_fc_layers, p_gnn, p_fc,
                 pretrained_concept_emb=None, diag_decompose=False, ablation=None, encoder_config={}):
        super().__init__()
        self.ablation = ablation
        self.model_name = model_name
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = RGCNNet(num_concepts, num_relations, num_basis, self.encoder.sent_dim, concept_dim, concept_in_dim, freeze_ent_emb,
                               num_gnn_layers, num_attention_heads, fc_dim, num_fc_layers, p_gnn, p_fc,
                               pretrained_concept_emb=pretrained_concept_emb, diag_decompose=diag_decompose, ablation=ablation)

    def forward(self, *inputs, layer_id=-1):
        """
        sent_vecs: (batch_size, d_sent)
        concept_ids: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)

        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension

        *lm_inputs, concept_ids, node_type_ids, adj_lengths, adj = inputs
        if 'no_lm' not in self.ablation:
            sent_vecs, _ = self.encoder(*lm_inputs, layer_id=layer_id)
        else:
            sent_vecs = torch.ones((bs * nc, self.encoder.sent_dim), dtype=torch.float).to(concept_ids.device)
        logits, attn = self.decoder(sent_vecs=sent_vecs, concepts=concept_ids, adj=adj, adj_lengths=adj_lengths)
        #logits = logits.view(bs, nc)
        return logits, attn


class LMRGCNDataLoader(object):

    def __init__(self, train_statement_path, train_label_path, train_adj_path, train_origin_path,
                 dev_statement_path, dev_label_path, dev_adj_path, dev_origin_path,
                 test_statement_path, test_adj_path, test_origin_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=256,
                 is_inhouse=False, inhouse_train_qids_path=None, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_statement_path, train_label_path, train_origin_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_statement_path, dev_label_path, dev_origin_path, model_type, model_name, max_seq_length, format=format)
        # if test_statement_path is not None:
        #     self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)

        self.num_choice = self.train_data[0].size(1)

        *train_extra_data, self.train_adj_data, n_rel = load_adj_data(train_adj_path, max_node_num, self.num_choice)
        self.train_data += train_extra_data
        *dev_extra_data, self.dev_adj_data, n_rel = load_adj_data(dev_adj_path, max_node_num, self.num_choice)
        self.dev_data += dev_extra_data
        assert all(len(self.train_qids) == len(self.train_adj_data) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data) == x.size(0) for x in [self.dev_labels] + self.dev_data)

        # pre-allocate an empty batch adj matrix
        self.adj_empty = torch.zeros((self.batch_size, self.num_choice, n_rel, max_node_num, max_node_num), dtype=torch.float32, device=device)
        self.eval_adj_empty = torch.zeros((self.eval_batch_size, self.num_choice, n_rel, max_node_num, max_node_num), dtype=torch.float32, device=device)

        # if test_statement_path is not None:
        #     *test_extra_data, self.test_adj_data, n_rel = load_adj_data(test_adj_path, max_node_num, self.num_choice)
        #     self.test_data += test_extra_data
        #     assert all(len(self.test_qids) == len(self.test_adj_data) == x.size(0) for x in [self.test_labels] + self.test_data)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r', encoding='utf-8') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return AdjDataBatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels,
                                     tensors=self.train_data, adj_empty=self.adj_empty, adj_data=self.train_adj_data)

    def train_eval(self):
        return AdjDataBatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels,
                                     tensors=self.train_data, adj_empty=self.eval_adj_empty, adj_data=self.train_adj_data)

    def dev(self):
        return AdjDataBatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels,
                                     tensors=self.dev_data, adj_empty=self.eval_adj_empty, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return AdjDataBatchGenerator(self.device, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels,
                                         tensors=self.train_data, adj_empty=self.eval_adj_empty, adj_data=self.train_adj_data)
        else:
            return AdjDataBatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels,
                                         tensors=self.test_data, adj_empty=self.eval_adj_empty, adj_data=self.test_adj_data)
