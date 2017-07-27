
# coding: utf-8

# In[1]:


import json
import glob
import pickle
import numpy as np
import MeCab
import sw
import sys
from gensim.models import word2vec

# In[2]:

wvsize = 200
#m = MeCab.Tagger()

# In[3]:

def corpusGenerator(fplist,train=False,past = 0):
    results = []
    add = results.append
    for fp in fplist:
        data= json.load(open(fp))
        logs = data['turns']
        for i in range(-1,len(logs),2):
            if train:
                annt = [a["breakdown"] for a in logs[i+1]["annotations"]]
                for an in annt:
                    probDist = list(map(int,[an=="O",an=="T",an=="X"]))
                    if i < 0:
                        add(("",logs[i+1]["utterance"],np.array(probDist)))
                    else:
                        add(("/".join(logs[j]["utterance"] for j in reversed(range(max(i-past,0),i+1))),logs[i+1]["utterance"],np.array(probDist)))
            else:
                annt = [a["breakdown"] for a in logs[i+1]["annotations"]]
                sm = len(annt)
                probDist = (annt.count("O")/sm,annt.count("T")/sm,annt.count("X")/sm)
                if i < 0:
                    add(("",logs[i+1]["utterance"],np.array(probDist)))
                else:
                    add(("/".join(logs[j]["utterance"] for j in reversed(range(max(i-past,0),i+1))),logs[i+1]["utterance"],np.array(probDist)))
    return results


# In[4]:


def tokenize(data):
    m = MeCab.Tagger("-Owakati")
    return [(m.parse(user).strip().split(),m.parse(system).strip().split(),probDist) for user,system,probDist in data]

"""
def w2vtrain(tokenized_trainData):
    rawCorpus = "" #wordEmbedding学習用
    for data in tokenized_trainData:
        rawCorpus += "\n" + " ".join(data[0])
        rawCorpus += "\n" + " ".join(data[1])
    rawCorpus = rawCorpus.strip()
    open("rawCorpus.txt","w").write(rawCorpus)
    sentences = word2vec.LineSentence("rawCorpus.txt")
    w2v = word2vec.Word2Vec(sentences,window=1)
    w2v.save("w2v.model")
    return w2v
"""


# In[7]:
"""
def vectorize(tokenizedData,w2v,boswe=False):
    #g = lambda x: list(filter(lambda word:word in w2v.wv.vocab.keys(), x))
    if boswe:
        vectorizer = sw.BOSWE()
        f = vectorizer.vectorize
    else:
        f = lambda x: np.array([w2v.wv[word].astype(np.float32) if word in w2v.wv.vocab.keys() else np.zeros(wvsize).astype(np.float32) for word in x])
    return np.array([(f(user),f(system),probDist) for user,system,probDist in tokenizedData])
"""

def vectorize(tokenizedData,w2v):
    #g = lambda x: list(filter(lambda word:word in w2v.wv.vocab.keys(), x))
    boswe = sw.BOSWE().vectorize
    f = lambda x: np.array([w2v.wv[word].astype(np.float32) if word in w2v.wv.vocab.keys() else np.zeros(wvsize).astype(np.float32) for word in x])
    return np.array([(f(user),f(system),probDist) for user,system,probDist in tokenizedData]),np.array([(boswe(user),boswe(system),probDist) for user,system,probDist in tokenizedData])

# In[8]:

def w2vtrain(c_dir = "../twitter/"):
    m = MeCab.Tagger("-Owakati")
    print("単語分割はじめるよ")
    sentences = [[word for word in m.parse(line).split()] for fp in glob.glob(c_dir+"corpus.txt") for line in open(fp)]
    print("単語分割おわったよ")
    print("word2vecの学習をはじめるよ")
    w2v = word2vec.Word2Vec(sentences,window=5,size=wvsize,min_count=5)
    print("w2v: vocabSize=",len(w2v.wv.vocab))
    w2v.save("w2v.model")
    print("word2vecの学習がおわったよ")
    return w2v

# In[9]:


if __name__=="__main__":
    print("コーパス作るよ")
    fp_rest = glob.glob('../projectnextnlp-chat-dialogue-corpus/json/rest1046/*.json')
    #fp_rest.extend(glob.glob('../projectnextnlp-chat-dialogue-corpus/json/rest1046/*.json'))
    fp_rest.extend(glob.glob('../DCM/*.json'))
    fp_rest.extend(glob.glob('../DIT/*.json'))
    fp_rest.extend(glob.glob('../IRS/*.json'))
    fp_rest.extend(glob.glob('../dev/*.json'))
    fp_init = glob.glob('../projectnextnlp-chat-dialogue-corpus/json/init100/*.json')
    trainData = corpusGenerator(fp_rest,True)
    validData = corpusGenerator(fp_init)
    tokenized_trainData = tokenize(trainData)
    tokenized_validData = tokenize(validData)
    #w2vpath='../BreakDialogue/word2vec.gensim.model'
    w2vpath='./w2v.model'
    if 'w2v' not in sys.argv:
        w2v = word2vec.Word2Vec.load(w2vpath)
    else:
        w2v = w2vtrain()
    
    vectorized_trainData,boswe_trainData = vectorize(tokenized_trainData,w2v)
    vectorized_validData,boswe_validData = vectorize(tokenized_validData,w2v)
    len(trainData)

    with open('corpus.pickle', mode='wb') as f:
        pickle.dump({
            "raw":{"train":trainData,"valid":validData},
            "tokenized":{"train":tokenized_trainData,"valid":tokenized_validData},
            "vectorized":{"train":vectorized_trainData,"valid":vectorized_validData},
            "clustering_feature":{"train":boswe_trainData,"valid":boswe_validData}
        },f)
    print("できたよ")
