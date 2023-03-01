import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import json
from src.use_kagnet.construct_graph import merged_relations

concept2id = None
id2concept = None
relation2id = None
id2relation = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def generate_triples_from_adj(adj_pk_path, mentioned_cpt_path, cpnet_vocab_path, triple_path):
    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    # with open(mentioned_cpt_path, 'r', encoding='utf-8') as fin:
    #     data = [json.loads(line) for line in fin]
    with open(mentioned_cpt_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    mentioned_concepts = [([concept2id[ac] for ac in item["answer_span_concept"] if ac in id2concept] + [concept2id[qc] for qc in item["sentence_span_concept"] if qc in id2concept]) for item in data]

    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)

    n_samples = len(adj_concept_pairs)
    triples = []
    mc_triple_num = []
    for idx, (adj_data, mc) in tqdm(enumerate(zip(adj_concept_pairs, mentioned_concepts)),
                                    total=n_samples, desc='loading adj matrices'):
        adj, concepts, _, _ = adj_data
        mapping = {i: (concepts[i]) for i in range(len(concepts))}  # index to corresponding grounded concept id
        ij = adj.row
        k = adj.col
        n_node = adj.shape[1]
        n_rel = 2 * adj.shape[0] // n_node
        i, j = ij // n_node, ij % n_node

        j = np.array([mapping[j[idx]] for idx in range(len(j))])
        k = np.array([mapping[k[idx]] for idx in range(len(k))])

        mc2mc_mask = np.isin(j, mc) & np.isin(k, mc)
        mc2nmc_mask = np.isin(j, mc) | np.isin(k, mc)
        others_mask = np.invert(mc2nmc_mask)
        mc2nmc_mask = ~mc2mc_mask & mc2nmc_mask
        mc2mc = i[mc2mc_mask], j[mc2mc_mask], k[mc2mc_mask]
        mc2nmc = i[mc2nmc_mask], j[mc2nmc_mask], k[mc2nmc_mask]
        others = i[others_mask], j[others_mask], k[others_mask]
        [i, j, k] = [np.concatenate((a, b, c), axis=-1) for (a, b, c) in zip(mc2mc, mc2nmc, others)]
        triples.append((i, j, k))
        mc_triple_num.append(len(mc2mc) + len(mc2nmc))

    with open(triple_path, 'wb') as fout:
        pickle.dump((triples, mc_triple_num), fout)
        print(f"Triples saved to {triple_path}")