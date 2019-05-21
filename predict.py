import json
import glob
from keras.models import Sequential
import keras
from keras.layers import Merge, LSTM, Dense,GRU
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.externals import joblib
import sys
import time
from utils import Preprocess, Ensemble,translate,EnsembleNomoto
from gensim.models import word2vec
import argparse



        
class Progress:
    def __init__(self,Max,domain):
        self.max = Max
        self.domain = domain
        self.current = 0
        self.now = None
        self.start = time.time()
    def __call__(self):
        self.current += 1
        self.now = time.time() - self.start
        pred = (self.max-self.current)*(self.now/self.current) if self.now != None else "unknown"
        sys.stdout.write("\r\033[K"+self.domain+" : "+str(self.current)+"/"+str(self.max)+"\t経過"+ ("%03.3f" % self.now)+
                         "秒\tあと"+ ("%03.3f" % pred) +"秒")
        sys.stdout.flush()
        if self.current == self.max:
            print()
        
        os.system("mkdir %s/%s"%(savedir,domain))
        
class Predictor:
    def __init__(self,namehead_keras='conv_kernel_attn',
                 n_clusters_keras=7,namehead_chainer='nomoto',n_clusters_chainer=5,pad_size=40):
        
        model_tkym_dir = './models/classifier/'  
        model_nmt_dir = './models/Series_LSTM/'  
        
        w2vpath='./models/w2v/w2v_512.model'
        w2v = word2vec.Word2Vec.load(w2vpath)
        self.pad_size = pad_size
        self.namehead_keras = namehead_keras
        self.namehead_chainer = namehead_chainer
        self.clf_tkym = Ensemble(namehead_keras,n_clusters_keras,method='mean',normalization='norm')
        self.clf_nmt = EnsembleNomoto(namehead_chainer,n_clusters_chainer,method='mean',normalization='norm')
        self.preprocess = Preprocess(w2vpath=w2vpath)
        
    def predict(self,user,system):
        user = [user] if type(user) == str else user
        system = [system] if type(system) == str else system
        user_token = list(map(self.preprocess.replace_lowfreq,user))
        sys_token = list(map(self.preprocess.replace_lowfreq,system))

        user_vecs = list(map(self.preprocess.vectorize,user_token))
        sys_vecs = list(map(self.preprocess.vectorize,sys_token))
        #sys_vecs_past = [[]] + sys_vecs[:-1]
        #print(len(user_vecs))

        user_vecs = sequence.pad_sequences(user_vecs,self.pad_size,dtype=np.float32)
        sys_vecs = sequence.pad_sequences(sys_vecs,self.pad_size,dtype=np.float32)
        #sys_vecs_past = sequence.pad_sequences(sys_vecs_past,pad_size,dtype=np.float32)


        probs = []
        if self.namehead_keras:
            probs.append(self.clf_tkym.ensemble(self.clf_tkym.predict([user_vecs,sys_vecs])))
        if self.namehead_chainer:
            probs.append(self.clf_nmt.ensemble(self.clf_nmt.predict([user_vecs.transpose(1,0,2),sys_vecs.transpose(1,0,2)])))
        probs = self.clf_tkym.ensemble(np.array(probs))

        breakdowns = translate(probs)
        print("breakdown: ",list(zip(user,system,breakdowns)))
        return list(zip(user,system,breakdowns,probs))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # parserを作る
    parser.add_argument('-k', '--keras') # オプションを追加します
    parser.add_argument('-n', '--kofkeras')
    parser.add_argument('-c', '--chainer') # 
    parser.add_argument('-m', '--kofchainer') # 
    parser.add_argument('-r', '--referencedir') # 
    parser.add_argument('-o', '--outputdir') # 
    args = parser.parse_args()
    print(args)

    n_clusters_keras = 0 if args.kofkeras is None else int(args.kofkeras)
    namehead_keras = args.keras
    n_clusters_chainer =0 if args.kofchainer is None else int(args.kofchainer)
    namehead_chainer = args.chainer

    
    predict = Predictor(
        n_clusters_keras = n_clusters_keras,
        namehead_keras = namehead_keras,
        n_clusters_chainer = n_clusters_chainer,
        namehead_chainer =  namehead_chainer,
        pad_size =40
    )
    while(1):
        predict.predict(input("user: "),input("system: "))
    """
    packed_results = {
        "dialogue-id":dataID,
        "turns":[
            {
                "turn-index":turnIndex, 
                "labels":[{"breakdown":breakdown, "prob-O":round(float(prob[0]),2), "prob-T":round(float(prob[1]),2), "prob-X":round(float(prob[2]),2)}]
            }
            for turnIndex, breakdown, prob in zip(turnIndexes, breakdowns, probs)
        ]
    }
    """ 