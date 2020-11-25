import numpy as np
import pickle
import scipy.sparse
import os
from sklearn.model_selection import train_test_split
import numpy.matlib

class opt():
    test_prop = 0.5

relation_base = '../../data/relation_pair/'

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()[:2]) for line in lines]
    return edges

def edges_to_matrix(edge_list):
    row_num, col_num = max([e[0] for e in edge_list]) + 1, \
                       max([e[1] for e in edge_list]) + 1
    mtx = np.matlib.zeros(shape=(row_num, col_num), dtype=np.float32)
    for (u, v) in edge_list:
        mtx[u, v] = 1
    return mtx

def edges_to_dicts(edge_list):
    edge_list.sort(key=lambda x: x[0])
    keys = [e[0] for e in edge_list]
    features = dict.fromkeys(keys, [])

    l = []
    key = keys[0]
    for (k, v) in edge_list:
        if k != key:
            features[key] = tuple(l)
            key = k
            l = []
        l.append(v)
    features[key] = tuple(l)
    return features

def jaccard(a, b):
    a, b = set(a), set(b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

user_api_pairs = read_edges_from_file(relation_base + 'pair_MS.txt')

train_user_api_pairs, test_user_api_pairs = train_test_split(user_api_pairs, test_size=opt.test_prop, random_state=42)
train_api_user_pairs = [(v, u) for u, v in train_user_api_pairs]
test_api_user_pairs = [(v, u) for u, v in test_user_api_pairs]
data_matrix = edges_to_matrix(user_api_pairs)
train_user_api_dict = edges_to_dicts(train_user_api_pairs)
test_user_api_dict = edges_to_dicts(test_user_api_pairs)
train_api_user_dict = edges_to_dicts(train_api_user_pairs)
test_api_user_dict = edges_to_dicts(test_api_user_pairs)

sim_users = np.zeros(shape=(data_matrix.shape[0], data_matrix.shape[0]))
sim_api = np.zeros(shape=(data_matrix.shape[1], data_matrix.shape[1]))

for u in train_user_api_dict.keys():
    for v in train_user_api_dict.keys():
        if u == v:
            continue
        sim_users[u][v] = jaccard(train_user_api_dict[u], train_user_api_dict[v])

for u in train_api_user_dict.keys():
    for v in train_api_user_dict.keys():
        if u == v:
            continue
        sim_api[u][v] = jaccard(train_api_user_dict[u], train_api_user_dict[v])

cache_path = './cache/'
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
if not os.path.exists('./results'):
    os.makedirs('./results')

with open('./cache/dataset.pkl', 'wb') as f:
    pickle.dump(train_user_api_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_user_api_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_api_user_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_api_user_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(scipy.sparse.csr_matrix(sim_users), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(scipy.sparse.csr_matrix(sim_api), f, pickle.HIGHEST_PROTOCOL)