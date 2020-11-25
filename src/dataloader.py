import numpy as np
import pickle
import random

data_path = '../cache/dataset.pkl'

with open(data_path, 'rb') as f:
    metapaths = [t.A for t in pickle.load(f)]


class DataLoader():
    def __init__(self, batch_size=32, train=True):
        with open(data_path, 'rb') as f:
            self.metapaths = [t.A for t in pickle.load(f)]
            self.user_features = pickle.load(f)
            self.trainset_dict = pickle.load(f)
            self.testset_dict = pickle.load(f)
            self.trainset_ms = pickle.load(f)
            self.train = train

        if train:
            self.uids = list(self.trainset_dict.keys())
            self.batch_size = batch_size
        else:
            self.uids = list(self.testset_dict.keys())
            self.batch_size = len(self.uids)

        self.cur_idx = 0
        self.meta_info = np.concatenate([m for m in self.metapaths], axis=1)
        self.available = True
        self.c_dim = max(len(self.user_features[i]['MC']) for i in self.uids)
        self.d_dim = max(len(self.user_features[i]['MD']) for i in self.uids)
        self.t_dim = max(len(self.user_features[i]['MT']) for i in self.uids)
        self.mask_dic = self.mask_set(self.trainset_dict)

    def mask_set(self, trainset_dict, r=0, item_n=1272):
        items = set(range(item_n))
        mask_dict = {}
        for k in trainset_dict.keys():
            rest = items.difference(set(trainset_dict[k]))
            masks = random.sample(rest, int(len(rest) * r))
            mask_dict[k] = masks
        return mask_dict

    def __iter__(self):
        return self

    def __next__(self):
        uids = self.uids[self.cur_idx: self.cur_idx + self.batch_size]

        ufeatures = []
        l = [[], [], []]
        for d, fea, idx in zip([self.c_dim, self.d_dim, self.t_dim], ['MC', 'MD', 'MT'], [0, 1, 2]):
            m = np.zeros([len(uids), d], np.int32)
            k = 0
            for i in uids:
                cates = self.user_features[i][fea]
                l[idx].append(len(cates))
                for j in range(len(cates)):
                    m[k][j] = cates[j]
                k += 1
            ufeatures.append(m)
        self.cur_idx += self.batch_size
        if self.cur_idx > len(self.uids):
            self.available = False
        meta_paths = [self.meta_info[i] for i in uids]
        if self.train:
            mask = np.ones([len(uids), 1272], np.float)
            ms = self.trainset_ms[uids]
            for i, v in enumerate(uids):
                zero_idxs = self.mask_dic[v]
                for j in zero_idxs:
                    mask[i][j] = 0
            return uids, ufeatures, meta_paths, l, mask, ms
        else:
            return uids, ufeatures, meta_paths, l

if __name__ == "__main__":
    pass