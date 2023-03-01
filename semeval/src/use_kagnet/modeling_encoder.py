import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, \
    XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers import OpenAIGPTTokenizer
from transformers.models.electra.modeling_electra import ElectraClassificationHead

GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'electra': list(ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
}

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}


def get_gpt_token_num():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    return len(tokenizer)


class ElectraPooler(nn.Module):
    def __init__(self, config):
        super(ElectraPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states

        assert not self.output_token_states or self.model_type in ('bert', 'roberta',)

        self.module = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.config = AutoConfig.from_pretrained(model_name)
        if self.model_type in ('electra',):
            self.pooler = ElectraPooler(self.config)

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
        elif self.model_type in ('electra',):
            sent_vecs = self.pooler(hidden_states)
        else:  # bert / roberta
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = self.module.pooler(hidden_states)
        return sent_vecs, all_hidden_states