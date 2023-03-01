import json
from tqdm import tqdm
from src.use_kagnet.construct_graph import merged_relations
from multiprocessing import Pool
import networkx as nx
import itertools
import pickle
import numpy as np
from scipy.sparse import coo_matrix

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def plain_graph_generation(qcs, acs, paths, rels):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple

    graph = nx.Graph()
    for p in paths:
        for c_index in range(len(p) - 1):
            h = p[c_index]
            t = p[c_index + 1]
            # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
            graph.add_edge(h, t, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        if cpnet_simple.has_edge(qc1, qc2):
            graph.add_edge(qc1, qc2, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            graph.add_edge(ac1, ac2, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc, ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    return nx.node_link_data(g)


def generate_subgraph(grounded_path, pruned_paths_path, cpnet_vocab_path, cpnet_graph_path, output_path):
    print(f'generating schema graphs for {grounded_path} and {pruned_paths_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    nrow = sum(1 for _ in open(grounded_path, 'r'))
    with open(grounded_path, 'r', encoding='utf-8') as fin_gr:
        mcps = json.load(fin_gr)
    with open(pruned_paths_path, 'r', encoding='utf-8') as fin_pf, open(output_path, 'w', encoding='utf-8') as fout:
        for mcp, line_pf in tqdm(zip(mcps, fin_pf), total=nrow):
            #mcp = json.loads(line_gr)
            qa_pairs = json.loads(line_pf)

            statement_paths = []
            statement_rel_list = []
            for qas in qa_pairs:
                if qas["pf_res"] is None:
                    cur_paths = []
                    cur_rels = []
                else:
                    cur_paths = [item["path"] for item in qas["pf_res"]]
                    cur_rels = [item["rel"] for item in qas["pf_res"]]
                statement_paths.extend(cur_paths)
                statement_rel_list.extend(cur_rels)

            qcs = [concept2id[c] for c in mcp["sentence_span_concept"] if c in id2concept]
            acs = [concept2id[c] for c in mcp["answer_span_concept"] if c in id2concept]

            gobj = plain_graph_generation(qcs=qcs, acs=acs,
                                          paths=statement_paths,
                                          rels=statement_rel_list)
            fout.write(json.dumps(gobj) + '\n')

    print(f'schema graphs saved to {output_path}')
    print()


def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


def concepts_to_adj_matrices_2hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        datas = json.load(fin)
    for data in datas:
        q_ids = set(concept2id[c] for c in data['sentence_span_concept'] if c in id2concept)
        a_ids = set(concept2id[c] for c in data['answer_span_concept'] if c in id2concept)
        q_ids = q_ids - a_ids
        qa_data.append((q_ids, a_ids))
    # with open(grounded_path, 'r', encoding='utf-8') as fin:
    #     for line in fin:
    #         dic = json.loads(line)
    #         q_ids = set(concept2id[c] for c in dic['qc'])
    #         a_ids = set(concept2id[c] for c in dic['ac'])
    #         q_ids = q_ids - a_ids
    #         qa_data.append((q_ids, a_ids))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair, qa_data), total=len(qa_data)))

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()