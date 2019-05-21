import json
import glob
import pickle
import numpy as np
import MeCab
from utils import Preprocess, ScalaList, ListDict
import pandas as pd

preprocess = Preprocess(w2vpath='./models/w2v/w2v_512.model')

fps = glob.glob('../projectnextnlp-chat-dialogue-corpus/json/rest1046/*.json')
fps.extend(glob.glob('../DCM/*.json'))
fps.extend(glob.glob('../DIT/*.json'))
fps.extend(glob.glob('../IRS/*.json'))
fps.extend(glob.glob('../dev/*.json'))
fps.extend(glob.glob('../projectnextnlp-chat-dialogue-corpus/json/init100/*.json'))

kind = ('O','T','X')
f = lambda x: np.eye(3)[kind.index(x)]

print("parsing json data...")
res_annt = []
res_dlg = []
annotators = set([annt['annotator-id'] for fp in fps for data in json.load(open(fp,'r'))['turns'] for annt in data['annotations']])
for fp in fps:
    datas = json.load(open(fp,'r'))
    a = datas
    #for data in datas['turns']:
    annotations = ListDict()
    dialogue = ListDict()
    for i,data in enumerate(datas['turns']):
        if i%2==1:
            continue
        if i >= 2:
            dialogue.append('user',datas['turns'][i-1]['utterance'])
        else:
            dialogue.append('user','')
        dialogue.append('system',datas['turns'][i]['utterance'])
        for annt in data['annotations']:
            annotations.append(annt['annotator-id'],f(annt['breakdown']))
            
    tmp = pd.DataFrame(datas['turns'])
    res_annt.append(pd.DataFrame(annotations))
    res_dlg.append(pd.DataFrame(dialogue))

print("generating dataframe objects:")
print("\tannotation data...",end="")
df_annotations = pd.concat(res_annt)
df_annotations = df_annotations.reset_index(drop=True)
print("Done")

print("\tdialogue data...",end="")
df_dialogue = pd.concat(res_dlg)
df_dialogue = df_dialogue.reset_index(drop=True)
print("Done")

print("\ttokenized data...",end="")
user_token = list(map(preprocess.replace_lowfreq,df_dialogue['user']))
sys_token = list(map(preprocess.replace_lowfreq,df_dialogue['system']))
print("Done")

print("\tvector data...",end="")
user_vecs = list(map(preprocess.vectorize,user_token))
sys_vecs = list(map(preprocess.vectorize,sys_token))
print("Done")

df_token = pd.DataFrame({'user':user_token,'system':sys_token})
df_vecs = pd.DataFrame({'user':user_vecs,'system':sys_vecs})

print("saving dataframe objects...")
df_annotations.to_pickle('./data/annotations.pickle')
df_dialogue.to_pickle('./data/dialogue.pickle')
df_token.to_pickle('./data/token.pickle')
df_vecs.to_pickle('./data/vecs.pickle')
print("complete!")