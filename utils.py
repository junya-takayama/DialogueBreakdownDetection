import json
import glob
import pickle
import numpy as np
import MeCab
from gensim.models import word2vec
from functools import reduce
import keras
from Series_LSTM import Series_LSTM
from chainer import serializers
from chainer import cuda


class Preprocess:
    def __init__(self,lf_path = './models/w2v/低頻度語.pickle',w2vpath='./models/w2v/w2v_512.model'):
        self.mc = MeCab.Tagger('mecabrc -F%m,%f[0]\\n')
        self.lowfreq = pickle.load(open(lf_path,'rb'))
        self.w2v = word2vec.Word2Vec.load(w2vpath)
    
    def replace_lowfreq(self,txt):
        txt = txt.replace(' ', '')
        return ['<<&%s&>>' % w.split(',')[1] if w.split(',')[0] in self.lowfreq or w.split(',')[0] not in self.w2v.wv.vocab.keys() else w.split(',')[0] 
                for w in self.mc.parse(txt).split('\n')[:-2]]

    def vectorize(self,tokenized):
        g = lambda x: list(filter(lambda word:word in self.w2v.wv.vocab.keys(), x))
        f = lambda x: np.array([self.w2v.wv[word] for word in x])
        return f(g(tokenized))
    
class ListDict(dict):
    #dict型を継承して，valueがlist型の場合の面倒な処理をメソッド化
    def __init__(self,d={}):
        dict.__init__(self,d)
        
    def append(self,key,elem):
        if key in self.keys():
            self[key].append(elem)
        else:
            self[key] = [elem]
            
    def extend(self,key,target):
        if key in self.keys():
            self[key].extend(target)
        else:
            self[key] = [target]

class ScalaList(list):
    def __init__(self,l=[]):
        list.__init__(self,l)
        
    def filter(self,f):
        return ScalaList(filter(f,self))
    
    def map(self,f):
        return ScalaList(map(f,self))
    
    def reduce(self,f):
        tmp = reduce(f,self)
        if type(tmp) == list:
            return ScalaList(tmp)
        else:
            return tmp

def evaluate(pred_proba,labels):
    pred = np.eye(3)[list(map(np.argmax,pred_proba))]
    seikai = np.eye(3)[list(map(np.argmax,labels))]
    tp = np.logical_and(pred,seikai)
    recall = tp.sum(axis=0) / seikai.sum(axis=0)
    precision = tp.sum(axis=0) / pred.sum(axis=0)
    fmeasure = 2*recall*precision/(recall+precision)
    accuracy = np.sum(tp)/len(pred)
    mse = np.sqrt(np.average((labels - pred_proba) ** 2))
    return fmeasure,recall,precision,accuracy,mse

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis=1).reshape(-1,1)

def norm(x):
    return x / np.sum(x,axis=1).reshape(-1,1)

def translate(preds):
    labels = ("O","T","X")
    f = lambda x: labels[np.argmax(x)]
    return [f(pred) for pred in preds]
    

class Ensemble:
    def __init__(self,namehead,n_clusters,method='max',normalization='norm',modeldir='./models/classifier/'):
        self.method = np.sum if method is 'average' else np.max
        self.normalization = softmax if normalization == 'softmax' else norm
        try:
            self.clf = keras.models.model_from_yaml(open(modeldir+namehead+'_k_'+str(n_clusters)+'.yaml'))
            self.weights = glob.glob(modeldir+namehead+'_k_'+str(n_clusters)+'_*.weight')
        except:
            print("keras models have not been loaded")
    def predict(self,x):
        res = []
        for weight in self.weights:
            self.clf.load_weights(weight)
            res.append(self.clf.predict(x))
        return res
    def ensemble(self,y):
        return self.normalization(self.method(y,axis=0))
        
class EnsembleNomoto:
    def __init__(self,namehead,n_clusters,method='max',normalization='norm',modeldir='./models/Series_LSTM/'):

        self.method = np.sum if method is 'average' else np.max
        self.normalization = softmax if normalization == 'softmax' else norm
        try:
            self.clf = Series_LSTM(
                parameters={
                    "hidden_size": 256,
                    "batch_size": 50,
                    "dr_ratio_sys": 0.2,
                }, 
                dr_ratio_sys=0.2)
            self.clf.to_gpu()
            self.weights = glob.glob(modeldir+namehead+'_k_'+str(n_clusters)+'_*.model')
        except:
            print("chainer models have not been loaded")
    def predict(self,x):
        res = []
        for weight in self.weights:
            serializers.load_hdf5(weight, self.clf)
            res.append(cuda.to_cpu(self.clf(x[0],x[1]).data))
        return res
    def ensemble(self,y):
        return self.normalization(self.method(y,axis=0))
    
