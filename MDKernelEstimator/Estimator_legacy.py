try:
    import tensorflow as tf
except:
    print('warning: tensorflow not found')
import numpy as np
import time
from MDKernelEstimator.gadgets import softmax


def self_distance(x, m, flag_tf=False):
    if len(x.shape)==1:
        x = np.reshape(x,(1,-1))
    if not flag_tf:
        dist = np.sum(np.matmul(x, m) * x, axis=1)
    else:
        dist = tf.reduce_sum(tf.matmul(x, m) * x, axis=1)
    # dist = np.array([np.matmul(np.matmul(xx.reshape(1, -1), m), xx.reshape(-1, 1)) for xx in x])
    return dist


def concat_arrays(a, a_inc):
    if (a is not None) and (a.size != 0):
        if len(a.shape) == 1:
            a_new = np.hstack([a, a_inc])
        else:
            a_new = np.vstack([a, a_inc])
    else:
        a_new = a_inc.copy()
    return a_new


class PredictResult:
    def __init__(self, y_test_predict=None, y_test_pdf=None, dist=None):
        self.__y_test_predict = y_test_predict
        self.__y_test_pdf = y_test_pdf
        self.__dist = dist
        self.__pdf_continuous = None
        self.__y_predict_continuous = None

    @property
    def y_test_predict(self):
        return self.__y_test_predict

    @property
    def y_test_pdf(self):
        return self.__y_test_pdf

    @property
    def dist(self):
        return self.__dist

    @property
    def pdf_continuous(self):
        return self.__pdf_continuous

    @property
    def y_predict_continuous(self):
        return self.__y_predict_continuous

    @pdf_continuous.setter
    def pdf_continuous(self, pdf_con):
        self.__pdf_continuous = pdf_con

    @y_predict_continuous.setter
    def y_predict_continuous(self, y_predict_con):
        self.__y_predict_continuous = y_predict_con


class Estimator:
    def __init__(self, x_train, y_train, k_n, m=None, cal_dist_mode=0):
        self.T, self.D = x_train.shape
        if m is None:
            m = np.eye(self.D)
        self._M = m
        self.KN = k_n
        self._X_train = x_train
        self._Y_train = y_train
        self.__y_train_predict = None
        self.__paras_train = None
        self.cal_dist_mode = cal_dist_mode
        self.__predict_prepared = False
        self.__predicted_within_train = False
        self.__predict_result = PredictResult()

    def reset_flags(self):
        self.__predict_prepared = False
        self.__predicted_within_train = False

    def formulate_train_paras(self, mat_x_train):
        vec_dist_train = self_distance(mat_x_train, self.m).reshape((1, -1))
        mat_transform_train = np.matmul(self.m, mat_x_train.T)
        return vec_dist_train, mat_transform_train

    def formulate_train_paras_self(self):
        vec_dist_train = self_distance(self.mat_x_train, self.m).reshape((1, -1))
        mat_transform_train = np.matmul(self.m, self.mat_x_train.T)
        return vec_dist_train, mat_transform_train

    @property
    def paras_train(self):
        if (self.__paras_train is None) or (not self.__predict_prepared):
            self.__paras_train = self.formulate_train_paras_self()
            self.__predict_prepared = True
        return self.__paras_train

    @property
    def mat_x_train(self):
        return self._X_train

    @property
    def mat_y_train(self):
        return self._Y_train

    @property
    def m(self):
        return self._M

    @property
    def predict_result(self):
        return self.__predict_result

    @mat_x_train.setter
    def mat_x_train(self, x_train):
        self._X_train = x_train
        self.T, self.D = self.mat_x_train.shape
        self.reset_flags()

    @mat_y_train.setter
    def mat_y_train(self, y_train):
        self._Y_train = y_train

    @m.setter
    def m(self, m):
        self._M = m
        self.reset_flags()

    def add_samples(self, x_train_inc, y_train_inc):
        self.mat_x_train = concat_arrays(self.mat_x_train, x_train_inc)
        self.mat_y_train = concat_arrays(self.mat_y_train, y_train_inc)
        paras_inc = self.formulate_train_paras(x_train_inc)
        self.__paras_train = tuple([np.hstack([self.__paras_train[i], paras_inc[i]]) for i in range(len(paras_inc))])

    def delete_samples(self, idx_delete):
        self.mat_x_train = np.delete(self.mat_x_train, idx_delete, axis=0)
        self.mat_y_train = np.delete(self.mat_y_train, idx_delete, axis=0)
        self.__paras_train = tuple([np.delete(t, idx_delete, axis=1) for t in self.__paras_train])

    def cal_dist(self, x_test):
        vec_dist_train, mat_transform_train = self.paras_train
        c = np.matmul(x_test, mat_transform_train)
        vec_dist_test = self_distance(x_test, self.m).reshape((1, -1))
        # dist = np.power(np.repeat(ad.T, s, axis=0) + np.repeat(bd, self.T, axis=1) - 2 * c, self.r_p)
        dist = vec_dist_train + vec_dist_test.T - 2 * c
        return dist

    def predict_from_dist(self, dist):
        if self.cal_dist_mode == 0:
            s = -dist
        elif self.cal_dist_mode == 1:
            s = 1 / dist
            s[np.isinf(dist)] = -float('inf')
        else:
            s = 1 / dist - dist
        y_test_pdf = np.array([softmax(self.KN * dt) for dt in s])
        y_test_predict = np.matmul(y_test_pdf, self._Y_train)
        return y_test_predict, y_test_pdf

    def predict(self, x_test):
        dist = self.cal_dist(x_test)
        y_test_predict, y_test_pdf = self.predict_from_dist(dist)
        self.__predict_result = PredictResult(y_test_predict, y_test_pdf, dist)
        return y_test_predict

    def predict_within_train(self, refresh=False):
        if (self.__y_train_predict is None) or (not self.__predicted_within_train) or refresh:
            dist = self.cal_dist(self.mat_x_train)
            dist += np.diag([np.inf] * self.T)
            self.__y_train_predict = self.predict_from_dist(dist)
            self.__predicted_within_train = True
        return self.__y_train_predict

    def get_pdf(self, h_w=0.1, num_div=10000, bandwidth=0.5):
        idx_sort_temp = np.argsort(self.mat_y_train)
        y_sorted = self.mat_y_train[idx_sort_temp]
        mat_pdf_sorted = self.predict_result.y_test_pdf[:,idx_sort_temp]
        vec_y_pdf = np.linspace(y_sorted[0] - bandwidth, y_sorted[-1] + bandwidth, num_div)
        mat_y_diff = y_sorted.reshape((-1,1)) - vec_y_pdf
        mat_exp = 1 / np.sqrt(2 * np.pi) / h_w * np.exp(-np.square(mat_y_diff) / (2 * h_w * h_w))
        self.predict_result.pdf_continuous = np.matmul(mat_pdf_sorted, mat_exp)
        self.predict_result.y_predict_continuous = vec_y_pdf
        return self.predict_result.y_predict_continuous, self.predict_result.pdf_continuous


class DistanceTrainer:
    def __init__(self, batch_size=100, epochs=100, kn_0=1, reg_alpha=1e-6, dict_para_optimizer=None, Tmax=None):
        if dict_para_optimizer is None:
            dict_para_optimizer = {'learning_rate': 1, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
        self.reg_alpha = reg_alpha
        self.dict_para_optimizer = dict_para_optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.epochs_show = min(1, int(epochs / 1000))
        self.KN0 = kn_0
        self.KN_base = None
        self.KN = None
        self.SubXTrain = None
        self.SubYTrain = None
        self.DistP = None
        self.Dist = None
        self.DistR = None
        self.DistRR = None
        self.ww_ = None
        self.w_ = None
        self.ws_ = None
        self.WS_ = None
        self.WMMMM_ = None
        self.WMMM_ = None
        self.WMM_ = None
        self.WM_ = None
        self.m_ = None
        self.AM = None
        self.Ad = None
        self.AD = None
        self.IMatch = None
        self.SubYtrainP = None
        self.loss = None
        self.reg_term = None
        self.op_tar = None
        self.optimizer = None
        self.train = None
        self.init_op = None
        self.sess = None
        self.opt = None
        self.r_m = None
        self.r_m_opt = None
        self.r_ww = None
        self.r_w = None
        self.r_ws = None
        self.r_WS = None
        self.r_WMMMM = None
        self.r_WMMM = None
        self.r_WMM = None
        self.r_WM = None
        self.r_loss = None
        self.r_SubYtrainP = None
        self.r_DistP = None
        self.r_Dist = None
        self.r_DistR = None
        self.r_DistRR = None
        self.r_IMatch = None
        self.r_KN = None
        self.r_KN_opt = None
        self.r_AD = None
        self.full_batch = False
        self.min_loss = float('inf')
        self.min_loss_epoch = 0
        self.cal_dist_mode = 0
        self.accuracies = {0: 0}
        self.loss_ts = {0: np.inf}
        self.Tmax = Tmax

    def gradient_descent(self, SubXTrain=None, SubYTrain=None):
        if self.full_batch:
            feed_dict = {}
        else:
            feed_dict = {
                self.SubXTrain: SubXTrain,
                self.SubYTrain: SubYTrain
                # self.train_phase: True
                # self.dropout_keep: self.keep
            }
        self.opt, self.r_loss, self.r_m, self.r_SubYtrainP, self.r_IMatch, self.r_Dist, self.r_DistR, \
        self.r_DistRR, self.r_DistP, self.r_KN, \
        self.r_AD, self.r_ww, self.r_w, self.r_ws, self.r_WS, self.r_WMMMM, self.r_WMMM, self.r_WMM, self.r_WM = \
            self.sess.run((self.train, self.loss, self.m_, self.SubYtrainP, self.IMatch, self.Dist, self.DistR,
                           self.DistRR, self.DistP, self.KN,
                           self.AD, self.ww_, self.w_, self.ws_, self.WS_, self.WMMMM_, self.WMMM_, self.WMM_,
                           self.WM_),
                          feed_dict=feed_dict)

        if np.any(self.r_Dist < 0):
            pass
            # print('illegal distance')

    def fit(self, mat_x_train, vec_y_train, rec_acc=False, ts0_init=0):
        size_train_temp, dim = mat_x_train.shape
        times = np.zeros(size_train_temp)
        self.init_issue(mat_x_train, vec_y_train)
        with tf.device('/cpu:0'):
            with tf.Session() as self.sess:
                self.sess.run(self.init_op)
                t0 = time.time()
                ts0 = ts0_init
                for epoch in range(1, self.epochs + 1):
                    if self.full_batch:
                        mat_x_train_t = None
                        vec_y_train_t = None
                    else:
                        # idx_sel = np.random.permutation(idx_all)[:self.batch_size]
                        idx_sel = np.argsort(times)[:self.batch_size]
                        mat_x_train_t = mat_x_train[idx_sel, :]
                        vec_y_train_t = vec_y_train[idx_sel]
                        times[idx_sel] += 1

                    self.gradient_descent(mat_x_train_t, vec_y_train_t)
                    loss_show = np.sqrt(self.r_loss * 2 / self.batch_size) * 1e3
                    if self.min_loss > loss_show:
                        self.min_loss = loss_show
                        self.min_loss_epoch = epoch
                        self.r_KN_opt = self.r_KN
                        self.r_m_opt = self.r_m
                    if (self.epochs_show <= 1) or (epoch % self.epochs_show == 0) or (epoch == self.epochs) or (
                            epoch == 1):
                        dt = time.time() - t0
                        print('Epoch %d: train: %.3f (min:%.3f of epoch %d) [%.2f s]' %
                              (epoch, loss_show, self.min_loss, self.min_loss_epoch, dt))
                        ts0 += dt
                        t0 = time.time()
                        if rec_acc:
                            aTrain_t = np.mean(1 - np.abs(vec_y_train_t - self.r_SubYtrainP) / self.r_SubYtrainP)
                            self.accuracies[ts0] = aTrain_t
                            self.loss_ts[ts0] = loss_show
                        if self.Tmax and ts0 > self.Tmax:
                            break
                        # acct = 100-((loss_show-3)/2+3)
                        # print('%.3f %.3f' % (acct, t0-t00))

    def init_issue(self, Xtrain, Ytrain):
        size_temp, dim = Xtrain.shape
        if self.batch_size < size_temp:
            self.SubXTrain = tf.placeholder(dtype=tf.float64, shape=(self.batch_size, dim))
            self.SubYTrain = tf.placeholder(dtype=tf.float64, shape=(self.batch_size,))
        else:
            self.full_batch = True
            self.batch_size = size_temp
            self.SubXTrain = tf.constant(Xtrain, dtype=tf.float64)
            self.SubYTrain = tf.constant(Ytrain, dtype=tf.float64)
        if self.r_ww is None:
            ww_init = np.zeros(dim)
        else:
            ww_init = self.r_ww
        self.ww_ = tf.Variable(ww_init, dtype=tf.float64)
        self.w_ = tf.nn.softmax(self.ww_)
        self.ws_ = tf.reshape(tf.sqrt(self.w_), (-1, 1))
        self.WS_ = tf.matmul(self.ws_, self.ws_, transpose_b=True)
        if self.r_WMMMM is None:
            WMMMM_init = np.zeros([dim, dim])
        else:
            WMMMM_init = self.r_WMMMM
        self.WMMMM_ = tf.Variable(WMMMM_init, dtype=tf.float64)
        self.WMMM_ = (tf.sigmoid(self.WMMMM_) - 0.5) * 2
        self.WMM_ = (self.WMMM_ + tf.transpose(self.WMMM_)) / 2
        self.WM_ = self.WMM_ - tf.diag(tf.diag_part(self.WMM_)) + tf.diag(np.ones(dim))

        # self.pp_ = tf.Variable(1, dtype=tf.float64)
        # self.p_ = tf.sigmoid(self.pp_) + 1
        # self.p_ = tf.pow(self.pp_, 2) + 0.01

        self.m_ = self.WS_ * self.WM_
        # self.m_ = tf.diag(tf.nn.softmax(tf.diag_part((self.mm_ + tf.transpose(self.mm_)) / 2)))
        self.AM = tf.matmul(tf.matmul(self.SubXTrain, self.m_), self.SubXTrain, transpose_b=True)
        self.Ad = tf.diag_part(self.AM)
        self.AD = tf.tile(tf.reshape(self.Ad, (1, -1)), [self.batch_size, 1])
        self.DistP = self.AD + tf.transpose(self.AD) - 2 * self.AM
        self.Dist = tf.cast(self.DistP, tf.float64)
        # self.Dist = tf.pow(self.DistP, self.p_)
        # self.Dist = self.AD + tf.transpose(self.AD) - 2 * self.AM
        if self.r_KN is None:
            init_kn = 1
        else:
            init_kn = self.r_KN
        # self.KN_base = tf.Variable(initial_value=np.log(init_kn), dtype=tf.float64)
        # self.KN = tf.exp(self.KN_base)
        self.KN = tf.Variable(initial_value=init_kn, dtype=tf.float64)
        self.DistR = self.Dist * self.KN * self.KN0 + tf.cast(tf.diag([np.inf] * self.batch_size), dtype=tf.float64)
        if self.cal_dist_mode == 0:
            self.DistRR = -self.DistR
        elif self.cal_dist_mode == 1:
            self.DistRR = tf.reciprocal(self.DistR) - tf.cast(tf.diag([np.inf] * self.batch_size), dtype=tf.float64)
        elif self.cal_dist_mode == 2:
            self.DistRR = tf.sigmoid(-self.DistR)
        else:
            self.DistRR = tf.reciprocal(self.DistR) - self.DistR

        self.IMatch = tf.transpose(tf.nn.softmax(self.DistRR))
        #
        self.SubYtrainP = tf.reshape(tf.matmul(tf.reshape(self.SubYTrain, (1, -1)), self.IMatch), (self.batch_size,))
        self.loss = tf.nn.l2_loss(tf.subtract(self.SubYTrain, self.SubYtrainP))
        self.reg_term = tf.reduce_sum(self.m_) * self.reg_alpha
        self.op_tar = self.loss + self.reg_term
        self.optimizer = tf.train.AdamOptimizer(**self.dict_para_optimizer)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=10, beta1=0.5, beta2=0.8, epsilon=1e-8)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1,)
        self.train = self.optimizer.minimize(self.op_tar)
        self.init_op = tf.global_variables_initializer()
