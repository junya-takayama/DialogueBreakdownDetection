# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cupy as cp
import sys
xp = cp
from keras.preprocessing import sequence
from chainer import cuda, optimizers, optimizer, using_config, serializers
from preprocess_chainer import mk_minibatch
from os import path
import os
from pprint import pprint
from collections import deque
from sklearn.cluster import KMeans
import time
pad_size = 40

class Trainer:
    def __init__(self,
                 model_classes, parameters, df_data='data/vecs.pickle', df_label='./data/annotations.pickle',
                 max_epoch=50, batch_size=50, cluster_num=1,namehead='nomoto',pad_size=40,emb_size=512):
        self.model_classes = model_classes
        self.params = parameters
        """
        self.df_vec = pd.read_pickle(df_data)
        self.sys_X = sequence.pad_sequences(self.df_vec['system'].values,pad_size,dtype=np.float32)
        self.usr_X = sequence.pad_sequences(self.df_vec['user'].values,pad_size,dtype=np.float32)
        idx = np.arange(len(self.usr_X))
        self.sys_X_train = self.sys_X[idx % 7 != 0]
        self.usr_X_train = self.usr_X[idx % 7 != 0]
        self.sys_X_val = self.sys_X[idx % 7 == 0]
        self.usr_X_val = self.usr_X[idx % 7 == 0]
        """
        

        
        
        self.df_annotations = pd.read_pickle(df_label)
        self.df_vecs = pd.read_pickle(df_data)
        users = self.df_vecs['user'].tolist()
        systems = self.df_vecs['system'].tolist()
        kmeans = KMeans(n_clusters=cluster_num,verbose=1)

        annt_dist = [(annt,self.df_annotations[annt].mean(),self.df_annotations[annt].count()) for annt in self.df_annotations.keys()]

        df_dist = pd.DataFrame(annt_dist,columns=['annt_id','annt_dist','annt_count'])

        ignore_thre_min = 33
        ignore_thre_max = 20000
        annotators = list(df_dist[(df_dist.annt_count >= ignore_thre_min) & (df_dist.annt_count <= ignore_thre_max)]['annt_id'])
        kmeans_feature = np.array(list(df_dist[(df_dist.annt_count >= ignore_thre_min) & (df_dist.annt_count <= ignore_thre_max)]['annt_dist']))
        clusters = kmeans.fit_predict(kmeans_feature)

        self.annt_clusters = [list(map(lambda x:x[1],filter(lambda x:x[0] == i, zip(clusters,annotators)))) for i in range(cluster_num)]
        
        #tmp = pd.read_pickle(df_label).reset_index(drop=True).fillna(0).sum(axis=1).values
        #self.label = xp.array([list(i) for i in tmp]) / xp.array([[xp.sum(x)] * 3 for x in tmp])
        #self.label_train = self.label[idx % 7 != 0]
        #self.label_val = self.label[idx % 7 == 0]
        self.emb_size = emb_size
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.namehead = namehead

        self.cluster_num = cluster_num

    def __call__(self, *args, **kwargs):
        for i,annt_ids in enumerate(self.annt_clusters):
            
            target_annt = self.df_annotations[annt_ids].dropna(how='all')
            indexes = list(target_annt.index)
            vecs = self.df_vecs.iloc[indexes,:]
            label_train = np.array([np.mean(list(filter(lambda x:type(x) != str,annts)),axis=0) for annts in target_annt.fillna('nan').values])
            usr_X_train = sequence.pad_sequences(vecs['user'],pad_size,dtype=np.float32)
            sys_X_train= sequence.pad_sequences(vecs['system'],pad_size,dtype=np.float32)
            tmp = list(zip(usr_X_train,sys_X_train,label_train))
            np.random.shuffle(tmp)
            usr_X_train,sys_X_train,label_train = map(np.array,zip(*tmp))
            
            del tmp
            del vecs
            del target_annt
            
            print("%dth cluster of %d" % (i, self.cluster_num),"data length :",len(label_train))
            for model_class, param in zip(self.model_classes, self.params):
                print("\tmodel: ", model_class)
                print("\tparams: ")
                self.train_one_model(model_class, param, xp.array(usr_X_train), xp.array(sys_X_train), xp.array(label_train), i)

    def train_one_model(self, model_class, parameter, usr, system, label, c_n):
        batch_size = parameter["batch_size"] if "batch_size" in parameter.keys() else self.batch_size
        model = None
        out_dir = None
        splitpoint = - int(len(usr) * 0.15)
        usr_X_val = usr[splitpoint:]
        sys_X_val = system[splitpoint:]
        label_val = label[splitpoint:]
        usr = usr[:splitpoint]
        system = system[:splitpoint]
        label = label[:splitpoint]
        
        if model_class == "Series_LSTM":
            from Series_LSTM import Series_LSTM
            pprint(parameter)
            model = Series_LSTM(emb_size=self.emb_size, parameters=parameter)
            out_dir = path.join("models", model_class)
            if not path.isdir(out_dir):
                os.system("mkdir -p %s" % out_dir)
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
            opt.add_hook(optimizer.Lasso(0.000005))
            os.system("echo 'epoch,loss' > %s" % (path.join(out_dir, "learning_curve_train.csv")))
            os.system("echo 'epoch,loss' > %s" % (path.join(out_dir, "learning_curve_val.csv")))
            os.system("echo 'epoch: %d, min_loss: %.4f' > %s" % (-1, min_loss, path.join(out_dir, "min_loss.txt")))
            
            
            for epoch in range(self.max_epoch):
                opt.new_epoch()
                np.random.seed()
                np.random.shuffle(idx)
                batch_num = len(usr) // batch_size
                start = time.time()
                for num in range(batch_num):

                    usr_minibatch = usr[idx[num * batch_size: (num + 1) * batch_size]].transpose(1,0,2)
                    sys_minibatch = system[idx[num * batch_size: (num + 1) * batch_size]].transpose(1,0,2)
                    label_minibatch = label[idx[num * batch_size: (num + 1) * batch_size]]
                    currenttime = time.time() - start
                    sys.stdout.write(
                        ("\r\033[Kepoch %d: train " % epoch) + 
                        str(int(currenttime)) + "秒経過/ あと" +
                        str((currenttime / (num + 1)) * (batch_num - num + 1)) + "秒"
                    )
                    sys.stdout.flush()
                    opt.update(model.forward, usr_minibatch, sys_minibatch, label_minibatch,
                               save_file=path.join(out_dir, "learning_curve_train.csv"), epoch=epoch)
                print()

                # エポックごとにモデルの評価
                
                with using_config('train', False):
                    losses = []
                    for num in range(len(sys_X_val) // batch_size):
                        usr_minibatch = usr_X_val[idx[num * batch_size: (num + 1) * batch_size]].transpose(1,0,2)
                        sys_minibatch = sys_X_val[idx[num * batch_size: (num + 1) * batch_size]].transpose(1,0,2)
                        label_minibatch = label_val[idx[num * batch_size: (num + 1) * batch_size]]
                        # if usr_minibatch is not None:
                        losses.append(cuda.to_cpu(model.forward(usr_minibatch, sys_minibatch, label_minibatch).data))
                # print(losses, type(losses))
                loss = np.mean(losses)
                print("\t\tepoch %d: val" % epoch,"loss:",loss)
                os.system("echo '%d,%.4f' >> %s" % (epoch, loss, path.join(out_dir, "learning_curve_val.csv")))
                if min_loss > loss:
                    serializers.save_hdf5(path.join(out_dir, self.namehead+'_k_'+str(self.cluster_num)+'_'+str(c_n)+'.model'), model)
                    min_loss = loss
                    os.system("echo 'epoch: %d, min_loss: %.4f' > %s" % (
                        epoch, min_loss, path.join(out_dir, "min_loss.txt")
                    ))
                    update_count = 0
                else:
                    update_count += 1
                    if update_count > 4:
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
                 cluster_num = int(sys.argv[1]),
                 parameters=[
                     {
                         "hidden_size": 256,
                         "batch_size": 128,
                         "dr_ratio_sys": 0.2,
                     }
                 ],
                 namehead=sys.argv[2])
    tr()
