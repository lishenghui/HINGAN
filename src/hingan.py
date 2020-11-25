import sys
sys.path.append("..")
import tensorflow as tf
import logging
import numpy as np
from dataloader import DataLoader
import utils
import pickle
import multiprocessing
import argparse
from tqdm import tqdm
cores = multiprocessing.cpu_count()

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--emb_dim', type=int, default=128, help='Dimension of feature embedding')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--n_cate', type=int, default=372, help='The number of categories')
    parser.add_argument('--n_des', type=int, default=99, help='The number of descriptions')
    parser.add_argument('--n_tag', type=int, default=2086, help='The number of tags')
    parser.add_argument('--n_item', type=int, default=1272, help='The number of items')
    parser.add_argument('--n_users', type=int, default=6958, help='The number of items')
    parser.add_argument('--d_epochs', type=int, default=30, help='The number of d model epoch')
    parser.add_argument('--g_epochs', type=int, default=30, help='The number of g model epoch')
    parser.add_argument('--n_metapaths', type=int, default=8, help='The number of metapaths')
    parser.add_argument('--lr', type=float, default=10e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.05, help=' beta1')
    parser.add_argument('--negative_ratio', type=float, default=0.5, help='negative ratio')
    parser.add_argument('--alpha', type=float, default=0.05, help='alpha')
    parser.add_argument('--out_path', type=str, default="../results/recgan_.txt", help='alpha')
    parser.add_argument('--weight', type=str, default="yes", help='alpha')

    return parser.parse_args()

opt = parse_args()
loader = DataLoader(train=False)
def simple_test_one_user(x):
    id, lst, idx = x[0], x[1], x[2]
    global loader
    if id in loader.trainset_dict.keys():
        for i in loader.trainset_dict[id]:
            lst[i] = 0
    recom = np.argsort(-lst)
    gnd = set(loader.testset_dict[id])
    pos = np.zeros(opt.n_item)
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

class Generator(object):
    def __init__(self, param=None):
        self.embed = tf.placeholder(tf.float32, shape=[None, opt.emb_dim * 3], name='embed')
        self.ms = tf.placeholder(tf.float32, [None, opt.n_item], name='ms')

        if param:
            d_layer1 = tf.layers.dense(self.embed, 128, kernel_initializer=param[0], bias_initializer=param[1], activation=tf.nn.relu, name='d1')
            d_layer2 = tf.layers.dense(d_layer1, 256, kernel_initializer=param[2], bias_initializer=param[3], activation=tf.nn.relu, name='d2')
            d_layer3 = tf.layers.dense(d_layer2, 512, kernel_initializer=param[4], bias_initializer=param[5], activation=tf.nn.relu, name='d3')
            self.gen_fea = tf.layers.dense(d_layer3, opt.n_item, kernel_initializer=param[6], bias_initializer=param[7],
                                           activation=tf.nn.relu, name='d_out')
        else:
            d_layer1 = tf.layers.dense(self.embed, 128, activation=tf.nn.relu, name='d1')
            d_layer2 = tf.layers.dense(d_layer1, 256, activation=tf.nn.relu, name='d2')
            d_layer3 = tf.layers.dense(d_layer2, 512, activation=tf.nn.relu, name='d3')
            self.gen_fea = tf.layers.dense(d_layer3, opt.n_item, activation=tf.nn.relu, name='d_out')

        self.masked_gen = tf.multiply(self.ms, self.gen_fea)
        self.g_params = []
        for scope in ['d1', 'd2', 'd3', 'd_out']:
            with tf.variable_scope(scope, reuse=True):
                self.g_params.append(tf.get_variable('kernel'))
                self.g_params.append(tf.get_variable('bias'))

    def save_model(self, sess, filename):
        param = sess.run(self.g_params)
        pickle.dump(param, open(filename, 'wb'))

class Discriminator(object):
    def __init__(self):
        self.input_r = tf.placeholder(tf.float32, shape=[None, opt.n_metapaths * opt.n_item], name='input_r')
        self.input_f = tf.placeholder(tf.float32, shape=[None, opt.n_item], name='input_f')
        self.mc = tf.placeholder(tf.int32, shape=[None, 6], name='mc')
        self.md = tf.placeholder(tf.int32, shape=[None, 3], name='md')
        self.mt = tf.placeholder(tf.int32, shape=[None, 6], name='mt')
        self.lc = tf.placeholder(tf.int32, [None,], name='lc')
        self.ld = tf.placeholder(tf.int32, [None,], name='ld')
        self.lt = tf.placeholder(tf.int32, [None,], name='lt')
        self.mask = tf.placeholder(tf.float32, [None, opt.n_item], name='mask')
        self.ms = tf.placeholder(tf.float32, [None, opt.n_item], name='ms')
        L = 256
        M = 128
        N = 64
        # Attention layer
        self.W0 = tf.get_variable('w0', [opt.n_item, 1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        self.att = tf.get_variable('weight', [8, 1], initializer=tf.truncated_normal_initializer(mean=0.2, stddev=0.5))
        # Hidden layer variables
        self.W2 = tf.get_variable('w1', [opt.emb_dim * 3 + opt.n_item, L], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        self.B2 = tf.Variable(tf.zeros([L]))
        self.W3 = tf.get_variable('w2', [L, M], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        self.B3 = tf.Variable(tf.zeros([M]))
        self.W4 = tf.get_variable('w3', [M, N], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        self.B4 = tf.Variable(tf.zeros([N]))
        self.W5 = tf.get_variable('w4', [N, 1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        self.B5 = tf.Variable(tf.zeros([1]))

        # Embedding variables
        self.emb_mc = tf.get_variable("emb_mc", [opt.n_cate + 1, opt.emb_dim], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        self.emb_md = tf.get_variable("emb_md", [opt.n_des + 1, opt.emb_dim], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        self.emb_mt = tf.get_variable("emb_mt", [opt.n_tag + 1, opt.emb_dim], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        self.i_emb = self.embedding_layer()

        # For real data training
        input_r = tf.reshape(self.input_r, shape=[-1, 8, 1272])
        if opt.weight=="yes":
            print('weight')
            self.weights = tf.tile(tf.nn.softmax(self.att, axis=0), [1, opt.n_item]) # [batch, 8]
        else:
            print('unweight')
            self.weights = tf.tile(tf.nn.softmax(self.att, axis=1), [1, opt.n_item]) / 8 # [batch, 8]
        self.input_nor = input_r / tf.tile(tf.reshape(tf.reduce_max(tf.maximum(input_r, 1), axis=2), [-1, 8, 1]), [1, 1, opt.n_item])
        self.atten_features = tf.reduce_sum(self.weights * self.input_nor, axis=1)
        # self.atten_features = self.atten_features / tf.tile(tf.reshape(tf.reduce_max(self.atten_features, axis=1), [-1, 1]), [1, opt.n_item])
        self.Y_r = self.construct(self.i_emb, self.atten_features)
        self.Y_f = self.construct(self.i_emb, self.input_f)

    def embedding_layer(self):
        features = []
        for param, sl, idx in zip([self.emb_mc, self.emb_md, self.emb_mt], [self.lc, self.ld, self.lt], [self.mc, self.md, self.mt]):
            h_emb = tf.nn.embedding_lookup(param, idx)
            mask = tf.sequence_mask(sl, tf.shape(h_emb)[1], dtype=tf.float32)  # [B, T]
            mask = tf.expand_dims(mask, -1)  # [B, T, 1]
            mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B, T, H]
            h_emb *= mask  # [B, T, H]
            hist = h_emb
            hist = tf.reduce_sum(hist, 1)
            hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(tf.maximum(sl, 1), 1), [1, tf.shape(hist)[1]]), tf.float32))
            features.append(hist)
        embed = tf.concat(features, axis=1)
        return embed

    def construct(self, embed, inv):
        features_f = tf.concat([embed, inv], 1)
        net = tf.nn.relu(tf.matmul(features_f, self.W2) + self.B2)
        net = tf.nn.relu(tf.matmul(net, self.W3) + self.B3)
        net = tf.nn.sigmoid(tf.matmul(net, self.W4) + self.B4)
        out = tf.reduce_sum(tf.matmul(net, self.W5), axis=1) + self.B5
        return out

logging.basicConfig(level=logging.DEBUG)
class GAN(object):
    def __init__(self):
        print("building GAN model...")
        self.d_model = None
        self.g_model = None

        self.build_discriminator()
        self.build_generator()
        self.best = 0

    def init_graph(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()

    def build_generator(self):
        """initializing the generator"""
        with tf.variable_scope("generator"):
            self.g_model = Generator()
    def build_discriminator(self):
        """initializing the discriminator discriminator"""
        with tf.variable_scope("discriminator"):
            self.d_model = Discriminator()

    def get_train_data_fake(self, uids, ufeatures, l):
        labels = np.array([0] * len(uids))
        embed = self.sess.run(self.d_model.i_emb,
                              feed_dict={self.d_model.mc: ufeatures[0], self.d_model.md: ufeatures[1],self.d_model.mt: ufeatures[2],
                                         self.d_model.lc: l[0],self.d_model.ld: l[1], self.d_model.lt: l[2]})
        gen_invs = self.sess.run(self.g_model.gen_fea, feed_dict={self.g_model.embed: embed})
        return uids, gen_invs, labels, embed

    def zr(self, x, negative_sample):
        negative_loss = tf.multiply(x, negative_sample)
        negative_loss = tf.pow(negative_loss, 2)
        negative_loss = tf.reduce_sum(negative_loss, 1, keepdims=True)
        return negative_loss

    def sample_negative_item(self, sample_R):
        negative_num = int(opt.negative_ratio * opt.n_item)
        negative_R = np.zeros((sample_R.shape[0], sample_R.shape[1]))
        for y, i in enumerate(sample_R):
            inde = np.where(i == 0)
            index = inde[0]
            item_idx = np.random.choice(index, negative_num)
            negative_R[y, item_idx] = 1
        return negative_R


    def train(self):
        print('Start training')
        self.hist_holder = tf.placeholder(tf.float32, shape=self.d_model.atten_features.shape, name='hist')
        loss_att = tf.reduce_sum(tf.pow(self.d_model.atten_features - self.hist_holder, 2), axis=1)

        loss_r = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_model.Y_r), logits=self.d_model.Y_r)
        loss_f = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_model.Y_f), logits=self.d_model.Y_f)
        loss_d = (loss_f + loss_r) / 2 + opt.beta * loss_att

        optimizer = tf.train.GradientDescentOptimizer(opt.lr)
        # optimizer = tf.train.AdamOptimizer(opt.lr, beta1=opt.beta1)
        t_vars = tf.trainable_variables()
        op_d = [var for var in t_vars if 'discriminator' in var.name]
        d_updates = optimizer.minimize(loss_d, var_list=op_d)
        negative_sample = tf.placeholder(tf.float32, shape=[None, opt.n_item], name='negative_sample')
        logit_g = self.d_model.construct(self.g_model.embed, self.g_model.masked_gen)
        zr_loss = self.zr(self.g_model.gen_fea, negative_sample)


        loss_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logit_g), logits=logit_g) + opt.alpha * tf.reduce_mean(zr_loss)
        op_g = [var for var in t_vars if 'generator' in var.name]
        g_updates = optimizer.minimize(loss_g, var_list=op_g)

        self.init_graph()
        for epoch in range(10):
            # D-step
            with tqdm(range(opt.d_epochs)) as pbar:
                loss_ = float('inf')
                for d_epoch in range(opt.d_epochs):
                    pbar.set_description("loss %f" % np.mean(loss_))
                    pbar.update()
                    for uid, ufeature, meta_info, l, mask, ms in DataLoader(batch_size=opt.batch_size):
                        if len(uid) == 0:
                            break
                        _, gen_features, labels_f, embed = self.get_train_data_fake(uid, ufeature, l)
                        _, loss, input_nor, y_r, y_f, atten, weight = self.sess.run([d_updates, loss_d, self.d_model.input_nor,
                                                        self.d_model.Y_r, self.d_model.Y_f, self.d_model.atten_features, self.d_model.weights],
                                                  feed_dict={
                                                      self.d_model.mc: ufeature[0],
                                                      self.d_model.md: ufeature[1],
                                                      self.d_model.mt: ufeature[2],
                                                      self.d_model.lc: l[0],self.d_model.ld: l[1],self.d_model.lt: l[2],
                                                      self.d_model.input_r: meta_info,
                                                      self.d_model.input_f: gen_features,
                                                      self.d_model.mask: mask,
                                                      self.hist_holder : ms
                                                  })
                        loss_ = loss

            # G-step
            with tqdm(range(opt.g_epochs)) as pbar:
                loss_ = float(0.)
                p3 = 0
                for g_epoch in range(opt.g_epochs):
                    pbar.set_description("loss %f, p3 %f" % (loss_, p3))
                    pbar.update()
                    for uid, ufeature, meta_info, l, _, ms in DataLoader(batch_size=opt.batch_size):
                        if len(uid) == 0:
                            break
                        negative_R = self.sample_negative_item(ms)
                        _, gen_features, _, embed = self.get_train_data_fake(uid, ufeature, l)
                        loss, _, = self.sess.run([loss_g, g_updates], feed_dict={self.g_model.embed : embed,
                                                                                 negative_sample :  negative_R,
                                                                                 self.g_model.ms : ms})
                        loss_ = np.mean(loss)
                    p3 = self.test_g_model()


    def test_g_model(self):
        loader = DataLoader(train=False, batch_size=200)
        results_raw = []
        for uids, ufeatures, meta_info, l in loader:
            if len(uids) == 0:
                break
            embed, atten, weight = self.sess.run([self.d_model.i_emb, self.d_model.atten_features, self.d_model.weights],
                                         feed_dict={self.d_model.mc: ufeatures[0], self.d_model.md: ufeatures[1],
                                                    self.d_model.mt: ufeatures[2],
                                                    self.d_model.lc: l[0], self.d_model.ld: l[1], self.d_model.lt: l[2],
                                                    self.d_model.input_r: meta_info})
            gen_invs = self.sess.run(self.g_model.gen_fea, feed_dict={self.g_model.embed: embed})
            param = list(zip(uids, gen_invs, range(len(uids)))) #+ [loader.trainset_dict, loader.testset_dict]
            results_raw.extend([simple_test_one_user(x) for x in param])

        results = np.round(np.mean(results_raw, axis=0), 4).reshape(-1, 4)
        if results[1][1] > self.best:
            self.best = results[1][1]
            np.set_printoptions(precision=4, suppress=True)
            np.savetxt(opt.out_path, results, fmt='%.4f')
            # self.g_model.save_model(self.sess, "gan_generator.pkl")
            np.set_printoptions(precision=4, suppress=True)

        return results[0][1]


if __name__ == "__main__":
    gan = GAN()
    gan.train()