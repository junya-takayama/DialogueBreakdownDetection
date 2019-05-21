import os
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import glob
import sys
from sklearn.cluster import KMeans
import utils

wvdim=512

if __name__ == "__main__":
    n_clusters = int(sys.argv[1])
    namehead = sys.argv[2]
    
    df_annotations = pd.read_pickle('./data/annotations.pickle')
    df_vecs = pd.read_pickle("./data/vecs.pickle")
    users = df_vecs['user'].tolist()
    systems = df_vecs['system'].tolist()
    kmeans = KMeans(n_clusters=n_clusters)

    annt_dist = [(annt,df_annotations[annt].mean(),df_annotations[annt].count()) for annt in df_annotations.keys()]

    df_dist = pd.DataFrame(annt_dist,columns=['annt_id','annt_dist','annt_count'])

    ignore_thre_min = 33
    ignore_thre_max = 20000
    annotators = list(df_dist[(df_dist.annt_count >= ignore_thre_min) & (df_dist.annt_count <= ignore_thre_max)]['annt_id'])
    kmeans_feature = np.array(list(df_dist[(df_dist.annt_count >= ignore_thre_min) & (df_dist.annt_count <= ignore_thre_max)]['annt_dist']))
    clusters = kmeans.fit_predict(kmeans_feature)

    pad_size = 40
    annt_clusters = [list(map(lambda x:x[1],filter(lambda x:x[0] == i, zip(clusters,annotators)))) for i in range(n_clusters)]

    try:
        os.mkdir('./models/classifier')
    except FileExistsError:
        print('directory already exist')

    for i,annt_ids in enumerate(annt_clusters):
        target_annt = df_annotations[annt_ids].dropna(how='all')
        indexes = list(target_annt.index)
        vecs = df_vecs.iloc[indexes,:]
        labels = np.array([np.mean(list(filter(lambda x:type(x) != str,annts)),axis=0) for annts in target_annt.fillna('nan').values])
        
        diff_indexes = np.array(df_annotations.drop(indexes).index)
        np.random.shuffle(diff_indexes)
        padding = max(5000 - len(indexes),0)
        diff_annt = df_annotations.iloc[diff_indexes[:padding],:]
        diff_vecs = df_vecs.iloc[diff_indexes[:padding],:]
        diff_labels = np.array([np.mean(list(filter(lambda x:type(x) != str,annts)),axis=0) for annts in diff_annt.fillna('nan').values])
        
        #if len(diff_labels) > 0:
            #vecs = pd.concat([vecs,diff_vecs])
            #labels = np.r_[labels,diff_labels]
        #labels = np.eye(3)[np.argmax(labels,axis=1)]
        user_x = sequence.pad_sequences(vecs['user'],pad_size,dtype=np.float32)
        system_x = sequence.pad_sequences(vecs['system'],pad_size,dtype=np.float32)
        #system_xp = sequence.pad_sequences(vecs['system-1'],pad_size,dtype=np.float32)
        tmp = list(zip(user_x,system_x,labels))
        np.random.shuffle(tmp)
        user_x,system_x,labels = map(np.array,zip(*tmp))
        ratio = np.average(labels,axis=0)
        inverseratio = 1/ratio
        inverseratio /= np.sum(ratio)
        print(ratio,inverseratio)
