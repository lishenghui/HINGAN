import numpy as np
import pickle
import scipy.sparse


class opt():
    test_prop = 0.5


from sklearn.model_selection import train_test_split

import os.path

relation_base = '../data/relation_pair/'


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


def read_user_features(path=relation_base):
    edge_list = read_edges_from_file(path + 'pair_MS.txt')
    keys = [e[0] for e in edge_list]
    features = dict.fromkeys(keys, {})
    for k in keys:
        features[k] = dict.fromkeys(['MC', 'MD', 'MT'], {})
    for n in ['MC', 'MD', 'MT']:
        edge_list = read_edges_from_file(path + 'pair_{0}.txt'.format(n))
        l = []
        key = keys[0]
        for (k, v) in edge_list:
            if k != key:
                features[key][n] = tuple(l)
                key = k
                l = []
            l.append(v)
        features[key][n] = tuple(l)
    return features


ms_edges = read_edges_from_file(relation_base + 'pair_MS.txt')
trainset, testset = train_test_split(ms_edges, test_size=opt.test_prop, random_state=42)
user_features = read_user_features()
trainset_dict = edges_to_dicts(trainset)
testset_dict = edges_to_dicts(testset)

ms = edges_to_matrix(trainset)
mc = edges_to_matrix(read_edges_from_file(relation_base + 'pair_MC.txt'))
md = edges_to_matrix(read_edges_from_file(relation_base + 'pair_MD.txt'))
mt = edges_to_matrix(read_edges_from_file(relation_base + 'pair_MT.txt'))
sc = edges_to_matrix(read_edges_from_file(relation_base + 'pair_SC.txt'))
sd = edges_to_matrix(read_edges_from_file(relation_base + 'pair_SD.txt'))
sp = edges_to_matrix(read_edges_from_file(relation_base + 'pair_SP.txt'))
st = edges_to_matrix(read_edges_from_file(relation_base + 'pair_ST.txt'))

l1 = ms * ms.T * ms
l2 = mt * mt.T * ms
l3 = mc * mc.T * ms
l4 = md * md.T * ms
l5 = ms * st * st.T
l6 = ms * sc * sc.T
l7 = ms * sd * sd.T
l8 = ms * sp * sp.T

metapaths = [l1, l2, l3, l4, l5, l6, l7, l8]
metapaths = [scipy.sparse.csr_matrix(t) for t in metapaths]

cache_path = '../cache/'
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
if not os.path.exists('../results'):
    os.makedirs('../results')
with open('../cache/dataset.pkl', 'wb') as f:
    pickle.dump(metapaths, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(user_features, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(trainset_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(testset_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(np.array(ms), f, pickle.HIGHEST_PROTOCOL)