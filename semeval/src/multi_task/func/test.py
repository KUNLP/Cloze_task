from transformers import ElectraTokenizer
from src.multi_task.func.multitask_utils_final import standard_sentence
import random
from random import shuffle
import numpy as np
import json
import pandas as pd
import pickle

#tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

#print(tokenizer.tokenize('smashwords'))
# example = "1839년 바그너는 괴테의 파우스트을 처음 읽고 그 내용에 마음이 끌려 이를 소재로 해서 하나의 교향곡을 쓰려는 뜻을 갖는다. 이 시기 바그너는 1838년에 빛 독촉으로 산전수전을 다 걲은 상황이라 좌절과 실망에 가득했으며 메피스토펠레스를 만나는 파우스트의 심경에 공감했다고 한다. 또한 파리에서 아브네크의 지휘로 파리 음악원 관현악단이 연주하는 베토벤의 교향곡 9번을 듣고 깊은 감명을 받았는데, 이것이 이듬해 1월에 파우스트의 서곡으로 쓰여진 이 작품에 조금이라도 영향을 끼쳤으리라는 것은 의심할 여지가 없다. 여기의 라단조 조성의 경우에도 그의 전기에 적혀 있는 것처럼 단순한 정신적 피로나 실의가 반영된 것이 아니라 베토벤의 합창교향곡 조성의 영향을 받은 것을 볼 수 있다. 그렇게 교향곡 작곡을 1839년부터 40년에 걸쳐 파리에서 착수했으나 1악장을 쓴 뒤에 중단했다. 또한 작품의 완성과 동시에 그는 이 서곡(1악장)을 파리 음악원의 연주회에서 연주할 파트보까지 준비하였으나, 실제로는 이루어지지는 않았다. 결국 초연은 4년 반이 지난 후에 드레스덴에서 연주되었고 재연도 이루어졌지만, 이후에 그대로 방치되고 말았다. 그 사이에 그는 리엔치와 방황하는 네덜란드인을 완성하고 탄호이저에도 착수하는 등 분주한 시간을 보냈는데, 그런 바쁜 생활이 이 곡을 잊게 한 것이 아닌가 하는 의견도 있다."
# print(example.find("교향곡"))

# with open('../../../data/multitask_dataset_train.json', 'r', encoding='utf8') as f:
#     multi_task_dataset_train = json.load(f)
#
# for con in multi_task_dataset_train[:5]:
#     print(con)
#     print()

# with open('../../../data/save_pickle_train_data.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# input_ids = data['total_input_ids']
# attention_mask = data['total_attention_mask']
# pair_ids = data['total_pair_ids']
# plausible_label = data['total_plausible_label']
# exist_label = data['total_relation_exist_label']
# type_label = data['total_relation_type_label']

tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
print(tokenizer.tokenize("mousse"))
# for sentence in input_ids:
#     print(tokenizer.decode(sentence))
# for idx, (pair, input_id, exist, typ) in enumerate(zip(pair_ids, input_ids, exist_label, type_label)):
#     if idx == 10:
#         break
#     for token in pair:
#         print(tokenizer.decode(input_id[token[0][0]:token[0][1]]))
#         print(tokenizer.decode(input_id[token[1][0]:token[1][1]]))
