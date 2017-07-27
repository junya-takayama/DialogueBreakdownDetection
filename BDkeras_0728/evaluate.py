
# coding: utf-8

# In[36]:


import preprocess
import json
import glob
from keras.models import Sequential
import keras
from keras.layers import Merge, LSTM, Dense,GRU
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional
import numpy as np
import pickle
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.externals import joblib
from BDkeras import Classifier
import sys

# In[47]:


#pklpath = "./models/test10.pickle"
pklpath = sys.argv[1]
name = pklpath.split("/")[-1].replace(".pickle","")
#pklpath = sys.argv[1]
pkl = pickle.load(open(pklpath,"rb"))
#pad_size = pkl["pad_size"]
pad_size = 30
pcapath = pkl["pcapath"]
knnpath = pkl["knnpath"]
pca = joblib.load(pcapath)
knn = joblib.load(knnpath)
from gensim.models import word2vec
w2vpath='./w2v.model'
w2v = word2vec.Word2Vec.load(w2vpath)
clf = Classifier(pad_size=pad_size,wvdim=200).decoder


# In[52]:


cluster_sent = [[] for i in range(10)]
res_v = []
cnt_l = {
    domain+"_"+str(i):0 for i in range(10) for domain in ["DCM","DIT","IRS"]
}
for domain in ["DCM","DIT","IRS"]:
    for fp in glob.glob("../DBDC2_ref/"+domain+"/*.json"):
        tmp = fp.split("/")
        dataID = tmp[-1].strip(".log.json")
        #domain = tmp[-2]
        result = {"dialogue-id":dataID,"turns":[]}
        aaa = preprocess.corpusGenerator([fp])
        rnn_vec,knn_vec= preprocess.vectorize(preprocess.tokenize(aaa),w2v=w2v)
        (xu_list,xs_list,y_list) = zip(*rnn_vec)
        (xu_cl,xs_cl,_) = zip(*knn_vec)
        xu_list = sequence.pad_sequences(xu_list,pad_size)
        xs_list = sequence.pad_sequences(xs_list,pad_size)
        #labels = knn.predict(np.c_[np.sum(xu_list,axis=1),np.sum(xs_list,axis=1)])
        #labels = knn.predict(np.average(xu_list,axis=1)-np.average(xs_list,axis=1))
        labels = knn.predict(pca.transform(xu_cl))
        #print(labels)
        for lbl in labels:
            cnt_l[domain+"_"+str(lbl)] += 1
        for i,zipped in enumerate(zip(aaa,xu_list,xs_list,labels)):
            bbb,xu,xs,label = zipped
            cluster_sent[label].append((bbb[:2],domain)) #あとで消す
            turnIndex = i*2
            clf.load_weights(pkl["modelpath"][label])
            predict = clf.predict([np.array([xu]),np.array([xs])])[0]
            probO,probT,probX = predict
            breakdown = ["O","T","X"][predict.argmax()]
            result["turns"].append({
                "turn-index":turnIndex,
                "labels":[{
                    "breakdown":breakdown,
                    "prob-O":float(probO),
                    "prob-T":float(probT),
                    "prob-X":float(probX),
                }]
            })
            res_v.append((bbb,probO,probT,probX))
        json.dump(result,open("./result/"+domain+"/"+dataID+".labels.json","w"))



