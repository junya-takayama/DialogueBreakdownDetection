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
import sys
import time
from utils import Preprocess, Ensemble,translate,EnsembleNomoto
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import argparse

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True
    )
)
set_session(tf.Session(config=config))



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

refdir = args.referencedir if args.referencedir is not None else "./data/DBDC2_ref/"
savedir = args.outputdir if args.outputdir is not None else "./result/"
os.system("mkdir %s"%savedir)

model_tkym_dir = './models/classifier/'  
model_nmt_dir = './models/Series_LSTM/'  
from gensim.models import word2vec
w2vpath='./models/w2v/w2v_512.model'
w2v = word2vec.Word2Vec.load(w2vpath)
clf_tkym = Ensemble(namehead_keras,n_clusters_keras,method='average',normalization='norm')
clf_nmt = EnsembleNomoto(namehead_chainer,n_clusters_chainer,method='average',normalization='norm')
preprocess = Preprocess(w2vpath=w2vpath)
pad_size = 40

        
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
        
        
for domain in ["DCM","DIT","IRS"]:
    os.system("mkdir %s/%s"%(savedir,domain))
    fps = glob.glob(refdir.rstrip("/")+"/"+domain+"/*.json")
    prog = Progress(len(fps),domain)
    for fp in fps:
        datas = json.load(open(fp,'r'))
        dataID = fp.split('/')[-1].strip(".log.json")
        prog()
        user = [""]
        user.extend([datas['turns'][i]['utterance'] for i in range(1,len(datas['turns']),2)])
        system = [datas['turns'][i]['utterance'] for i in range(0,len(datas['turns'])+1,2)]
        
        user_token = list(map(preprocess.replace_lowfreq,user))
        sys_token = list(map(preprocess.replace_lowfreq,system))
        
        user_vecs = list(map(preprocess.vectorize,user_token))
        sys_vecs = list(map(preprocess.vectorize,sys_token))
        #sys_vecs_past = [[]] + sys_vecs[:-1]
        #print(len(user_vecs))
            
        user_vecs = sequence.pad_sequences(user_vecs,pad_size,dtype=np.float32)
        sys_vecs = sequence.pad_sequences(sys_vecs,pad_size,dtype=np.float32)
        #sys_vecs_past = sequence.pad_sequences(sys_vecs_past,pad_size,dtype=np.float32)
        

        probs = []
        if namehead_keras:
            probs.append(clf_tkym.ensemble(clf_tkym.predict([user_vecs,sys_vecs])))
        if namehead_chainer:
            probs.append(clf_nmt.ensemble(clf_nmt.predict([user_vecs.transpose(1,0,2),sys_vecs.transpose(1,0,2)])))
        probs = clf_tkym.ensemble(np.array(probs))
            
        breakdowns = translate(probs)
        turnIndexes = [i for i in range(0,len(datas['turns'])+1,2)]
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
        json.dump(packed_results,open(savedir.rstrip("/")+"/"+domain+"/"+dataID+".labels.json","w"))
