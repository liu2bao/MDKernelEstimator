try:
    import tensorflow as tf
except:
    print('warning: tensorflow not found')
import numpy as np
import time
import pickle


def self_distance(x, m, flag_tf=False):
    if len(x.shape) == 1:
        x = np.reshape(x, (1, -1))
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


class ParaHolderOrth:
    def __init__(self, batch_size=100, epochs=100, kn_0=1, reg_alpha=1e-6, dict_para_optimizer=None, Tmax=None, tol=0):
        if dict_para_optimizer is None:
            dict_para_optimizer = {'learning_rate': 1, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
        self.reg_alpha = reg_alpha
        self.dict_para_optimizer = dict_para_optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.epochs_show = min(1, int(epochs / 1000))
        self.KN0 = kn_0
        self.r_m = None
        self.r_P = None
        self.r_m_opt = None
        self.r_wvecraw = None
        self.r_wvec = None
        self.r_loss = None
        self.r_loss_old = None
        self.r_my_loss = None
        self.r_Y_predict = None
        self.r_DistP = None
        self.r_Dist = None
        self.r_DistR = None
        self.r_DistRR = None
        self.r_IMatch = None
        self.r_KN = None
        self.r_KN_opt = None
        self.r_AD = None
        self.r_rotvecraw = None
        self.r_rotvec = None
        self.full_batch = False
        self.min_loss = float('inf')
        self.min_loss_epoch = 0
        self.cal_dist_mode = 0
        self.accuracies = {0: 0}
        self.loss_ts = {0: np.inf}
        self.Tmax = Tmax
        self.real_size = None
        self.use_test = False
        self.X_test_comp = None
        self.Y_test_comp = None
        self.tol = tol


class ParaHolderTFOrth:
    def __init__(self):
        self.KN = None
        self.SubXTrain = None
        self.SubYTrain = None
        self.SubXTest = None
        self.SubYTest = None
        self.Y_compare = None
        self.DistP = None
        self.Dist = None
        self.DistR = None
        self.DistRR = None
        self.wvecraw_ = None
        self.wvec_ = None
        self.P_ = None
        self.m_ = None
        self.rotvecraw_ = None
        self.rotvec_ = None
        self.AM = None
        self.Ad = None
        self.Bd = None
        self.AD = None
        self.BD = None
        self.IMatch = None
        self.Y_predict = None
        self.loss = None
        self.my_loss = None
        self.reg_term = None
        self.op_tar = None
        self.optimizer = None
        self.train = None
        self.init_op = None
        self.sess = None
        self.opt = None


class ParaSaver:
    def __init__(self, path_save=None):
        self._path_save = path_save

    @property
    def path_save(self):
        return self._path_save

    @path_save.setter
    def path_save(self, ps):
        self._path_save = ps

    def save_para(self, para):
        with open(self.path_save, 'wb') as f:
            pickle.dump(para, f)

    def load_para(self):
        with open(self.path_save, 'rb') as f:
            para = pickle.load(f)
        return para


class DistanceTrainerOrth:
    def __init__(self, batch_size=100, epochs=100, kn_0=1, reg_alpha=1e-6, dict_para_optimizer=None, Tmax=None, tol=0):
        self.ph = ParaHolderOrth(batch_size=batch_size, epochs=epochs, kn_0=kn_0, reg_alpha=reg_alpha,
                                 dict_para_optimizer=dict_para_optimizer, Tmax=Tmax, tol=tol)
        self.ph_tf = ParaHolderTFOrth()
        self.ps = ParaSaver()

    @property
    def cal_dist_mode(self):
        return self.ph.cal_dist_mode

    @property
    def batch_size(self):
        return self.ph.batch_size

    @cal_dist_mode.setter
    def cal_dist_mode(self,m):
        self.ph.cal_dist_mode = m

    @batch_size.setter
    def batch_size(self,bs):
        self.ph.batch_size = bs

    def gradient_descent(self, SubXTrain=None, SubYTrain=None):
        if self.ph.full_batch:
            feed_dict = {}
        else:
            feed_dict = {
                self.ph_tf.SubXTrain: SubXTrain,
                self.ph_tf.SubYTrain: SubYTrain
                # self.train_phase: True
                # self.dropout_keep: self.keep
            }
        self.ph_tf.opt, self.ph.r_loss, self.ph.r_my_loss, self.ph.r_m, self.ph.r_Y_predict, self.ph.r_IMatch, self.ph.r_Dist, \
        self.ph.r_DistR, self.ph.r_DistRR, self.ph.r_DistP, self.ph.r_KN, self.ph.r_AD, \
        self.ph.r_wvecraw, self.ph.r_wvec, self.ph.r_P, self.ph.r_rotvecraw, self.ph.r_rotvec = \
            self.ph_tf.sess.run(
            (self.ph_tf.train, self.ph_tf.loss, self.ph_tf.my_loss, self.ph_tf.m_, self.ph_tf.Y_predict, self.ph_tf.IMatch, self.ph_tf.Dist,
             self.ph_tf.DistR, self.ph_tf.DistRR, self.ph_tf.DistP, self.ph_tf.KN, self.ph_tf.AD,
             self.ph_tf.wvecraw_, self.ph_tf.wvec_, self.ph_tf.P_, self.ph_tf.rotvecraw_, self.ph_tf.rotvec_),
                feed_dict=feed_dict)

        if np.any(self.ph.r_Dist < 0):
            pass
            # print('illegal distance')

    def fit(self, mat_x_train, vec_y_train, rec_acc=False, ts0_init=0,
            mat_x_test=None, vec_y_test=None, epochs_save=None):
        size_train_temp, dim = mat_x_train.shape
        times = np.zeros(size_train_temp)
        self.init_issue(mat_x_train, vec_y_train, mat_x_test, vec_y_test)
        with tf.device('/cpu:0'):
            with tf.Session() as self.ph_tf.sess:
                self.ph_tf.sess.run(self.ph_tf.init_op)
                t0 = time.time()
                ts0 = ts0_init
                for epoch in range(1, self.ph.epochs + 1):
                    if self.ph.full_batch:
                        mat_x_train_t = None
                        vec_y_train_t = None
                    else:
                        # idx_sel = np.random.permutation(idx_all)[:self.batch_size]
                        idx_sel = np.argsort(times)[:self.batch_size]
                        mat_x_train_t = mat_x_train[idx_sel, :]
                        vec_y_train_t = vec_y_train[idx_sel]
                        times[idx_sel] += 1

                    self.gradient_descent(mat_x_train_t, vec_y_train_t)
                    loss_show = np.sqrt(self.ph.r_loss * 2 / self.ph.batch_size) * 1e3
                    if self.ph.min_loss > loss_show:
                        self.ph.min_loss = loss_show
                        self.ph.min_loss_epoch = epoch
                        self.ph.r_KN_opt = self.ph.r_KN
                        self.ph.r_m_opt = self.ph.r_m
                    if (self.ph.epochs_show <= 1) or (epoch % self.ph.epochs_show == 0) or (epoch == self.ph.epochs) or (
                            epoch == 1):
                        dt = time.time() - t0
                        print('Epoch %d: train: %.3f (min:%.3f of epoch %d) [%.2f s]' %
                              (epoch, loss_show, self.ph.min_loss, self.ph.min_loss_epoch, dt))
                        ts0 += dt
                        if rec_acc:
                            if self.ph.use_test:
                                y_comp_t = self.ph.Y_test_comp
                            else:
                                y_comp_t = vec_y_train_t
                            aTrain_t = np.mean(1 - np.abs(y_comp_t - self.ph.r_Y_predict) / self.ph.r_Y_predict)
                            self.ph.accuracies[ts0] = aTrain_t
                            self.ph.loss_ts[ts0] = loss_show
                        if self.ph.Tmax and ts0 > self.ph.Tmax:
                            break

                        if isinstance(epochs_save,int) and (epoch % epochs_save==0):
                            self.ps.save_para(self.ph)
                        # acct = 100-((loss_show-3)/2+3)
                        # print('%.3f %.3f' % (acct, t0-t00))
                        t0 = time.time()
                    if self.ph.r_loss_old:
                        if np.abs(self.ph.r_loss-self.ph.r_loss_old)<self.ph.tol:
                            break
                    self.ph.r_loss_old = self.ph.r_loss

    def init_issue(self, Xtrain, Ytrain, Xtest=None, Ytest=None):
        size_temp, dim = Xtrain.shape
        flag_test_exists = False
        size_test = None
        self.ph.real_size = size_temp
        if (Xtest is not None) and (Ytest is not None):
            flag_test_exists = True
            size_test = Xtest.shape[0]
            self.ph.use_test = True
            self.ph.X_test_comp = Xtest
            self.ph.Y_test_comp = Ytest
            self.ph.real_size = size_test
            self.ph_tf.SubXTest = tf.constant(Xtest, dtype=tf.float64)
            self.ph_tf.SubYTest = tf.constant(Ytest, dtype=tf.float64)

        if self.batch_size < size_temp:
            self.ph_tf.SubXTrain = tf.placeholder(dtype=tf.float64, shape=(self.batch_size, dim))
            self.ph_tf.SubYTrain = tf.placeholder(dtype=tf.float64, shape=(self.batch_size,))
        else:
            self.ph.full_batch = True
            self.batch_size = size_temp
            self.ph_tf.SubXTrain = tf.constant(Xtrain, dtype=tf.float64)
            self.ph_tf.SubYTrain = tf.constant(Ytrain, dtype=tf.float64)

        if self.ph.r_rotvecraw is None:
            rotvecraw_init = np.zeros(dim-1)
        else:
            rotvecraw_init = self.ph.r_rotvecraw

        self.ph_tf.rotvecraw_ = tf.Variable(rotvecraw_init, dtype=tf.float64)
        self.ph_tf.rotvec_ = tf.nn.sigmoid(self.ph_tf.rotvecraw_)*2*np.pi
        self.ph_tf.P_ = tf.constant(np.eye(dim), dtype=tf.float64)

        for d in range(dim-1):
            diag_temp = tf.constant(np.ones(dim), dtype=tf.float64)
            diat_temp_cos = \
                (tf.one_hot(d, dim, dtype=tf.float64)+tf.one_hot(d+1, dim, dtype=tf.float64)) * \
                (tf.cos(self.ph_tf.rotvec_[d]) - 1)
            diag_temp += diat_temp_cos
            Rsubt = tf.diag(diag_temp)

            ohmat_temp = tf.matmul(tf.reshape(tf.one_hot(d+1,dim,dtype=tf.float64),(dim,1)),
                                   tf.reshape(tf.one_hot(d,dim,dtype=tf.float64),(1,dim)))
            ohmat_temp += (-tf.transpose(ohmat_temp))
            Rsubt += (ohmat_temp * tf.sin(self.ph_tf.rotvec_[d]))
            self.ph_tf.P_ = tf.matmul(self.ph_tf.P_, Rsubt)

        if self.ph.r_wvecraw is None:
            wvecraw_init = np.zeros(dim)
        else:
            wvecraw_init = self.ph.r_wvecraw

        self.ph_tf.wvecraw_ = tf.Variable(wvecraw_init, dtype=tf.float64)
        self.ph_tf.wvec_ = tf.nn.softmax(self.ph_tf.wvecraw_)
        self.ph_tf.m_ = tf.matmul(tf.matmul(self.ph_tf.P_,tf.diag(self.ph_tf.wvec_)), self.ph_tf.P_, transpose_b=True)

        if flag_test_exists:
            self.ph_tf.Ad = tf.reduce_sum(tf.matmul(self.ph_tf.SubXTrain, self.ph_tf.m_) * self.ph_tf.SubXTrain, axis=1)
            self.ph_tf.Bd = tf.reduce_sum(tf.matmul(self.ph_tf.SubXTest, self.ph_tf.m_) * self.ph_tf.SubXTest, axis=1)
            self.ph_tf.AD = tf.tile(tf.reshape(self.ph_tf.Ad, (-1, 1)), [1, size_test])
            self.ph_tf.BD = tf.tile(tf.reshape(self.ph_tf.Bd, (1, -1)), [self.batch_size, 1])
            self.ph_tf.AM = tf.matmul(tf.matmul(self.ph_tf.SubXTrain, self.ph_tf.m_), self.ph_tf.SubXTest, transpose_b=True)
            self.ph_tf.DistP = self.ph_tf.AD + self.ph_tf.BD - 2 * self.ph_tf.AM
        else:
            self.ph_tf.AM = tf.matmul(tf.matmul(self.ph_tf.SubXTrain, self.ph_tf.m_), self.ph_tf.SubXTrain, transpose_b=True)
            self.ph_tf.Ad = tf.diag_part(self.ph_tf.AM)
            self.ph_tf.AD = tf.tile(tf.reshape(self.ph_tf.Ad, (1, -1)), [self.batch_size, 1])
            self.ph_tf.DistP = self.ph_tf.AD + tf.transpose(self.ph_tf.AD) - 2 * self.ph_tf.AM

        self.ph_tf.Dist = tf.cast(self.ph_tf.DistP, tf.float64)
        # self.Dist = tf.pow(self.DistP, self.p_)
        # self.Dist = self.AD + tf.transpose(self.AD) - 2 * self.AM
        if self.ph.r_KN is None:
            init_kn = 1
        else:
            init_kn = self.ph.r_KN
        # self.KN_base = tf.Variable(initial_value=np.log(init_kn), dtype=tf.float64)
        # self.KN = tf.exp(self.KN_base)
        self.ph_tf.KN = tf.Variable(initial_value=init_kn, dtype=tf.float64)
        self.ph_tf.DistR = self.ph_tf.Dist * self.ph_tf.KN * self.ph.KN0
        if not flag_test_exists:
            self.ph_tf.DistR += tf.cast(tf.diag([np.inf] * self.batch_size), dtype=tf.float64)

        if self.cal_dist_mode == 0:
            self.ph_tf.DistRR = -self.ph_tf.DistR
        elif self.cal_dist_mode == 1:
            self.ph_tf.DistRR = tf.reciprocal(self.ph_tf.DistR)
            if not flag_test_exists:
                self.ph_tf.DistR -= tf.cast(tf.diag([np.inf] * self.batch_size), dtype=tf.float64)
        elif self.cal_dist_mode == 2:
            self.ph_tf.DistRR = tf.sigmoid(-self.ph_tf.DistR)
        else:
            self.ph_tf.DistRR = tf.reciprocal(self.ph_tf.DistR) - self.ph_tf.DistR

        self.ph_tf.IMatch = tf.nn.softmax(self.ph_tf.DistRR, axis=0)
        # self.ph_tf.IMatch = tf.nn.softmax(self.ph_tf.DistRR)
        SubYTrain_vec = tf.reshape(self.ph_tf.SubYTrain, (1, -1))
        self.ph_tf.Y_predict = tf.matmul(SubYTrain_vec, self.ph_tf.IMatch)
        if flag_test_exists:
            self.ph_tf.Y_compare = tf.reshape(self.ph_tf.SubYTest, (1, -1))
        else:
            self.ph_tf.Y_compare = SubYTrain_vec
        self.ph_tf.loss = tf.nn.l2_loss(tf.subtract(self.ph_tf.Y_compare, self.ph_tf.Y_predict))
        self.ph_tf.my_loss = tf.sqrt(tf.reduce_mean(tf.square(self.ph_tf.Y_compare - self.ph_tf.Y_predict)))
        self.ph_tf.reg_term = tf.reduce_sum(self.ph_tf.m_) * self.ph.reg_alpha
        self.ph_tf.op_tar = self.ph_tf.loss + self.ph_tf.reg_term
        self.ph_tf.optimizer = tf.train.AdamOptimizer(**self.ph.dict_para_optimizer)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=10, beta1=0.5, beta2=0.8, epsilon=1e-8)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1,)
        self.ph_tf.train = self.ph_tf.optimizer.minimize(self.ph_tf.op_tar)
        self.ph_tf.init_op = tf.global_variables_initializer()
