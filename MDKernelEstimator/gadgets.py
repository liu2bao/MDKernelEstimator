import numpy as np
import matplotlib.pyplot as plt


# %%
def softmax(z):
    ez = np.exp(z)
    f = np.logical_and(np.isinf(ez),ez>0)
    if np.any(f):
        ez = np.zeros(z.shape)
        ez[f] = 1
        ez[~f] = 0
    out = ez/np.sum(ez)
    return out


def cal_entropy_item(p):
    if p == 0:
        s = 0
    else:
        s = -p*np.log(p)
    return s


def cal_entropy(p_distribute):
    s = np.sum([cal_entropy_item(p) for p in p_distribute])
    return s


def cal_entropy_mat(mat_pdf):
    s = np.sum(-mat_pdf * np.log(mat_pdf),axis=1)
    return s


def root_mean_square(dx):
    rms = np.sqrt(np.mean(np.square(dx)))
    return rms


#%%
def compare_real_predict(Xtrain,Ytrain,Xtest,Ytest,model,n=0,lw=0.5):
    model.fit(Xtrain, Ytrain)
    Yp = model.predict(Xtest)
    idx_arange = np.argsort(Ytest)
    figt = plt.figure(('test_%d' % n))
    figt.clf()
    axt = figt.gca()
    l1, = axt.plot(Ytest[idx_arange],lw=lw)
    l2, = axt.plot(Yp[idx_arange],lw=lw)
    axt.legend([l1,l2],['real','predict'])
    return model,Yp
