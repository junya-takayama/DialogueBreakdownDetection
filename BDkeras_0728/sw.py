
# coding: utf-8

# In[24]:


from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import MeCab
import numpy as np
import pickle

class BOSWE:
    
    def __init__(self,init=False,n=1000):
        self.m = MeCab.Tagger("-Owakati")
        self.n = n
        if init:
            self.init()
        else:
            try:
                tmp = pickle.load(open("boswe.model","rb"))
                self.cent = tmp["cent"]
                self.label2word = tmp["label2word"]
                self.word2label = tmp["word2label"]
            except:
                self.init()
    
    def init(self,w2vpath = "./w2v.model"):
        w2v = Word2Vec.load(w2vpath)
        words = []
        vecs =[]
        for word in w2v.wv.vocab.keys():
            words.append(word)
            vecs.append(w2v.wv[word])
        del w2v
        kmeans = KMeans(n_clusters=self.n,verbose=1,n_init=2)
        kmeans.fit_predict(vecs)
        self.cent = kmeans.cluster_centers_
        self.label2word = [[] for i in range(self.n)]
        self.word2label = {}
        for label,word in zip(kmeans.labels_,words):
            self.label2word[label].append(word)
            self.word2label[word] = label
        pickle.dump({
            "cent" : self.cent,
            "label2word" : self.label2word,
            "word2label" : self.word2label,            
        },open("boswe.model","wb"))
    
    def vectorize(self,seq):
        vec = np.zeros(self.n)
        for word in seq:
            try:
                vec[self.word2label[word]] += 1
            except:
                pass
        vec2 = np.array([self.cent[self.word2label[word]] if word in self.word2label.keys() else np.zeros(200) for word in seq])
        return vec.astype(np.float32)

