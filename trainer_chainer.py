# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cupy as cp
xp = cp
from keras.preprocessing import sequence
from chainer import cuda, optimizers, optimizer, using_config, serializers
from preprocess_chainer import mk_minibatch
from os import path, system
from pprint import pprint
from collections import deque

pad_size = 40

class Trainer:
    def __init__(self,
                 model_classes, parameters, df_data='data/vecs.pickle', df_label='./data/annotations.pickle',
                 max_epoch=50, batch_size=50, cluster_num=1,namehead='nomoto'):
        self.model_classes = model_classes
        self.params = parameters

        self.df_vec = pd.read_pickle(df_data)
        self.sys_X = sequence.pad_sequences(self.df_vec['system'].values,pad_size,dtype=np.float32)
        self.usr_X = sequence.pad_sequences(self.df_vec['user'].values,pad_size,dtype=np.float32)
        idx = np.arange(len(self.usr_X))
        self.sys_X_train = self.sys_X[idx % 7 != 0]
        self.usr_X_train = self.usr_X[idx % 7 != 0]
        self.sys_X_val = self.sys_X[idx % 7 == 0]
        self.usr_X_val = self.usr_X[idx % 7 == 0]
        self.emb_size = max(e.shape[-1] for e in self.sys_X)

        tmp = pd.read_pickle(df_label).reset_index(drop=True).fillna(0).sum(axis=1).values

        self.label = xp.array([list(i) for i in tmp]) / xp.array([[xp.sum(x)] * 3 for x in tmp])
        self.label_train = self.label[idx % 7 != 0]
        self.label_val = self.label[idx % 7 == 0]

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.namehead = namehead

        self.cluster_num = cluster_num

    def __call__(self, *args, **kwargs):
        for i in range(self.cluster_num):
            print("%dth cluster of %d" % (i, self.cluster_num))
            usr_X_train = self.usr_X_train
            sys_X_train = self.sys_X_train
            label_train = self.label_train
            # usr_X_train = self.usr_X_train[i]
            # sys_X_train = self.sys_X_train[i]
            # label_train = self.label_train[i]
            for model_class, param in zip(self.model_classes, self.params):
                print("\tmodel: ", model_class)
                print("\tparams: ")
                self.train_one_model(model_class, param, usr_X_train, sys_X_train, label_train, i)

    def train_one_model(self, model_class, parameter, usr, sys, label, c_n):
        batch_size = parameter["batch_size"] if "batch_size" in parameter.keys() else self.batch_size
        model = None
        out_dir = None
        if model_class == "Series_LSTM":
            from Series_LSTM import Series_LSTM
            pprint(parameter)
            model = Series_LSTM(emb_size=self.emb_size, parameters=parameter)
            out_dir = path.join("models", model_class)
            if not path.isdir(out_dir):
                system("mkdir -p %s" % out_dir)
                print("\t\tmake %s dir" % out_dir)

        if model:
            min_loss = np.inf
            update_count = 0
            cuda.get_device_from_id(0).use()
            model.to_gpu()
            idx = np.arange(len(usr))
            model.reset()
            opt = optimizers.Adam()
            opt.setup(model)
            opt.add_hook(optimizer.WeightDecay(0.0001))
            opt.add_hook(optimizer.GradientClipping(5))
            system("echo 'epoch,loss' > %s" % (path.join(out_dir, "learning_curve_train.csv")))
            system("echo 'epoch,loss' > %s" % (path.join(out_dir, "learning_curve_val.csv")))
            system("echo 'epoch: %d, min_loss: %.4f' > %s" % (-1, min_loss, path.join(out_dir, "min_loss.txt")))
            for epoch in range(self.max_epoch):
                opt.new_epoch()
                np.random.seed()
                np.random.shuffle(idx)
                print("\t\tepoch %d: train" % epoch)
                for num in range(len(usr) // batch_size):
                    usr_minibatch = mk_minibatch(
                        usr[idx[num * batch_size: (num + 1) * batch_size]], self.emb_size)
                    sys_minibatch = mk_minibatch(
                        sys[idx[num * batch_size: (num + 1) * batch_size]], self.emb_size)
                    # print(num, usr_minibatch.shape, sys_minibatch.shape)
                    label_minibatch = label[idx[num * batch_size: (num + 1) * batch_size]]
                    
                    # if usr_minibatch is not None:
                    print(model(usr_minibatch,sys_minibatch).data.shape)
                    opt.update(model.forward, usr_minibatch, sys_minibatch, label_minibatch,
                               save_file=path.join(out_dir, "learning_curve_train.csv"), epoch=epoch)

                # エポックごとにモデルの評価
                print("\t\tepoch %d: val" % epoch)
                with using_config('train', False):
                    losses = []
                    for num in range(len(self.sys_X_val) // batch_size):
                        usr_minibatch = mk_minibatch(
                            self.usr_X_val[num * batch_size: (num + 1) * batch_size], self.emb_size)
                        sys_minibatch = mk_minibatch(
                            self.sys_X_val[num * batch_size: (num + 1) * batch_size], self.emb_size)
                        # print(num, usr_minibatch.shape, sys_minibatch.shape)
                        label_minibatch = self.label_val[num * batch_size: (num + 1) * batch_size]
                        # if usr_minibatch is not None:
                        losses.append(cuda.to_cpu(model.forward(usr_minibatch, sys_minibatch, label_minibatch).data))
                # print(losses, type(losses))
                loss = np.mean(losses)
                system("echo '%d,%.4f' >> %s" % (epoch, loss, path.join(out_dir, "learning_curve_val.csv")))
                if min_loss > loss:
                    serializers.save_hdf5(path.join(out_dir, self.namehead+'_k_'+str(self.cluster_num)+'_'+str(c_n)+'.model'), model)
                    min_loss = loss
                    system("echo 'epoch: %d, min_loss: %.4f' > %s" % (
                        epoch, min_loss, path.join(out_dir, "min_loss.txt")
                    ))
                    update_count = 0
                else:
                    update_count += 1
                    if update_count > 6:
                        return 0


if __name__ == '__main__':
    """
    def parse(param):
        return {"hidden_size": int(param[0]), "batch_size": int(param[1]),
                "dr_ratio_sys": param[2]}
    hs_kinds = [64]
    bs_kinds = [50]
    sdr_kinds = [0.2]
    aa, bb, cc = np.meshgrid(hs_kinds, bs_kinds, sdr_kinds)
    params = [parse(p) for p in np.c_[aa.ravel(), bb.ravel(), cc.ravel()]]
    """
    tr = Trainer(model_classes=["Series_LSTM"], 
                 parameters=[
                     {
                         "hidden_size": 256,
                         "batch_size": 128,
                         "dr_ratio_sys": 0.2,
                     }
                 ],
                 namehead='nomoto')
    tr()
