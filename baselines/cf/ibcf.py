import pickle
import numpy as np
import utils
from tqdm import tqdm

with open('./cache/dataset.pkl', 'rb') as f:
    trainset_dict = pickle.load(f)
    testset_dict = pickle.load(f)
    hist_api_user_dict = pickle.load(f)
    _ = pickle.load(f)
    sim_users = pickle.load(f).todense()
    sim_apis = pickle.load(f).todense()

trainuserSet, testuserSet = set(trainset_dict.keys()), set(testset_dict.keys())
n_items = sim_apis.shape[0]


def cf_api(u,i,cf_sim):
    """
    计算给定user对api的打分
    :param u:       用户的id
    :param i:       api的id
    :param cf_sim:  api的相似度矩阵
    :return:        user对api的打分
    """
    K = 10
    # 找到调用过该api的所有用户一句相似度由高到低排序
    sim = [(l, cf_sim[i, l]) for l in trainset_dict.get(u, [])]
    sim.sort(key=lambda x: x[1], reverse=True)
    x = 0
    # 对于相似度为top-K以内的邻居，如果该邻居调用过该api，则本user与api的相似度加上
    # user与邻居的相似度。
    for s in sim[:K]:
        if s[0]==u:
            continue
        else:
            x = x+s[1]
    return x

def simple_test_one_user(id, reclist):
    """
    测试单个用户推荐的准确度， 结果是一个列表，包括top3, top5, top10的召回率、准确率、MRR、NDCG
    :param id: 用户id
    :param reclist: 生成的打分列表
    :return: 召回率、准确率、MRR、NDCG
    """
    global loader
    if id in trainset_dict.keys():
        for i in trainset_dict[id]:
            reclist[i] = 0
    recom = np.argsort(-reclist)
    gnd = set(testset_dict[id])
    pos = np.zeros(n_items)
    for i, e in enumerate(recom):
        if e in gnd:
            pos[i] = 1
    result = []
    for k in [3, 5, 10]:
        r = utils.recall_at_k(pos, k, len(gnd))
        p = utils.precision_at_k(pos, k)
        mrr = utils.mrr_at_k(pos, k)
        ndcg = utils.ndcg_at_k(pos, k)
        result.extend([r, p, mrr, ndcg])
    return result

results_raw = []
for user in tqdm(testuserSet):
    item_score = np.zeros(n_items)
    for i in range(n_items):
        x = cf_api(user, i, sim_apis)
        item_score[i] = x
    res = simple_test_one_user(user, item_score)
    items_score_pair = list(zip(range(len(hist_api_user_dict)), item_score))
    items_score_pair.sort(key=lambda x: x[1], reverse=True)
    results_raw.extend([res])

results = np.round(np.mean(results_raw, axis=0), 4).reshape(-1, 4)
print(results)
