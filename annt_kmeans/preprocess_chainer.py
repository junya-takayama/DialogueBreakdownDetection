# coding: utf-8
import json
import pickle
import MeCab
#from utils import Preprocess, ScalaList, ListDict
import pandas as pd
import numpy as np
import cupy as xp
# preprocess = Preprocess()

"""
def preprocessor(json_path):
    fps = [json_path]

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
    df_annotations = pd.concat(res_annt)
    df_annotations = df_annotations.reset_index(drop=True)
    df_dialogue = pd.concat(res_dlg)
    df_dialogue = df_dialogue.reset_index(drop=True)
    user_token = list(map(preprocess.replace_lowfreq,df_dialogue['user']))
    sys_token = list(map(preprocess.replace_lowfreq,df_dialogue['system']))
    user_vecs = list(map(preprocess.vectorize,user_token))
    sys_vecs = list(map(preprocess.vectorize,sys_token))
    df_token = pd.DataFrame({'user':user_token,'system':sys_token})
    df_vecs = pd.DataFrame({'user':user_vecs,'system':sys_vecs})
    return df_annotations, df_dialogue, df_token, df_vecs
"""

def mk_minibatch(minibatch, emb_size):
    if len(minibatch) > 0:
        max_size = max(len(e) for e in minibatch)+1
    else:
        return None
    pad = np.zeros(emb_size).astype("float32")
    new_minibatch = [np.array([pad for _ in range(max_size - len(e))]+list(e)) 
                     for e in minibatch]
    return np.array(new_minibatch).transpose(1, 0, 2)

