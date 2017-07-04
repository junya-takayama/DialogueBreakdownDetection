
# coding: utf-8

# In[525]:


from keras.models import Sequential
import keras
from keras.layers import Merge, LSTM, Dense,GRU, SimpleRNN, core
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from pprint import pprint
import sys
from sklearn.externals import joblib
from datetime import datetime as dt

# In[526]:
pad_size = 15

class Classifier:
    def __init__(self,n_class=3,batch_size=100,pad_size=30,wvdim=200):
        encoder_a = Sequential()
        encoder_a.add(GRU(output_dim=wvdim, batch_input_shape=(None, pad_size, wvdim), return_sequences=False,dropout_U=0.5))
        #encoder_a.add(GRU(100, input_shape=(timesteps, data_dim)))

        encoder_b = Sequential()
        encoder_b.add(GRU(output_dim=wvdim, batch_input_shape=(None, pad_size, wvdim), return_sequences=False,dropout_U=0.5))
        #encoder_b.add(GRU(100, input_shape=(timesteps, data_dim)))

        decoder = Sequential()
        decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
        decoder.add(core.Dropout(0.5))
        decoder.add(Dense(200, activation='relu'))
        decoder.add(core.Dropout(0.5))
        #decoder.add(Dense(300, activation='relu'))
        #decoder.add(core.Dropout(0.6))
        #decoder.add(Dense(100, activation='relu'))
        #decoder.add(core.Dropout(0.3))
        decoder.add(Dense(n_class, activation='softmax'))
        decoder.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['mean_squared_error'])
        self.decoder = decoder


# In[527]:


def clustering(train_user,train_system,valid_user,valid_system,y_train,y_valid,n=3):
    splitPoint = len(train_user)
    user = np.average(np.r_[train_user,valid_user],axis=1)
    system = np.average(np.r_[train_system,valid_system],axis=1)
    vec = system - user
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(vec)
    result_train_user,result_valid_user = dataSplit(train_user,valid_user,splitPoint,kmeans,n)
    result_train_system,result_valid_system = dataSplit(train_system,valid_system,splitPoint,kmeans,n)
    result_train_y,result_valid_y = dataSplit(y_train,y_valid,splitPoint,kmeans,n)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(vec,kmeans.labels_)
    return knn,result_train_user,result_train_system,result_valid_user,result_valid_system,result_train_y,result_valid_y

def dataSplit(train,valid,splitPoint,kmeans,n):
    result_train = [[] for i in range(n)]
    result_valid = [[] for i in range(n)]
    labels_train = kmeans.labels_[:splitPoint]
    labels_valid = kmeans.labels_[splitPoint:]
    for i,label in enumerate(labels_train):
        result_train[label].append(train[i])
    for i,label in enumerate(labels_valid):
        result_valid[label].append(valid[i])
    return result_train,result_valid


# In[528]:


def binarize(labels):
    return np.array([[0,0,1] if label[2] > 0.5 else [0,1,0] if label[1] > 0.5 else [1,0,0] for label in labels])


# In[529]:


def f_measure(predict,values=[1]):
    P = 0
    C = 0
    R = 0
    for pred,corr in zip(predict,y_val):
        if pred.argmax() in values:
            P += 1
        if corr.argmax() in values:
            C += 1
        if pred.argmax() in values and corr.argmax() in values:
            R += 1

    recall = R/P if P>0 else 0
    precision = R/C if C>0 else 0
    try:
        f = (2*recall*precision)/(recall+precision)
    except:
        f = 0
    return {"recall":recall,"precision":precision,"f_measure":f}


# In[530]:
if __name__ == "__main__":
    n = int(sys.argv[1])
    with open('corpus.pickle',mode='rb') as f:
        corpus = pickle.load(f)
    # generate dummy training data

    user,system,labels = zip(*corpus['vectorized']['train'])
    x_train_a_all = sequence.pad_sequences(user,pad_size,dtype=np.float32)
    #x_train_a = sequence.pad_sequences(system,pad_size,dtype=np.float32)
    x_train_b_all = sequence.pad_sequences(system,pad_size,dtype=np.float32)
    y_train_all = np.array(labels)

    # generate dummy validation data
    user,system,labels = zip(*corpus['vectorized']['valid'])
    x_val_a_all = sequence.pad_sequences(system,pad_size,dtype=np.float32)
    x_val_b_all = sequence.pad_sequences(user,pad_size,dtype=np.float32)
    y_val_all = np.array(labels)


    # In[531]:



    knn_model,train_user_cluster,train_system_cluster,valid_user_cluster,valid_system_cluster,train_y_cluster,valid_y_cluster     = clustering(x_train_a_all,x_train_b_all,x_val_a_all,x_val_b_all,y_train_all,y_val_all,n)


    # In[537]:
    #tdatatime = dt.now()
    #tstr = tdatatime.strftime("%Y%m%d%H%M%S")
    namehead = "./models/"+sys.argv[2]

    total = 0
    cnt = 0
    result = {
        "numOfClusters":n,
        "f_measures":{},
        "weightpath":[],
        "modelpath":[],
        "model":Classifier(pad_size=pad_size).decoder.to_json(),
        "knnpath":namehead+".knn",
        "pad_size":pad_size,
        "history":[]
    }

    for i in range(n):
        x_train_a = np.array(train_user_cluster[i])
        x_train_b = np.array(train_system_cluster[i])
        y_train = np.array(train_y_cluster[i])

        x_val_a = np.array(valid_user_cluster[i])
        x_val_b = np.array(valid_system_cluster[i])
        y_val = np.array(valid_y_cluster[i])

        print(knn_model.predict(x_train_b[0] - x_train_a[0]))
        
        decoder = Classifier(pad_size=pad_size).decoder
        es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        history = decoder.fit([x_train_a, x_train_b], y_train,
                    batch_size=256, nb_epoch=100,
                    validation_data=([x_val_a, x_val_b], y_val),shuffle=True)
        predict = decoder.predict([x_val_a,x_val_b])
        for pred,label in zip(predict,y_val):
            total += 1
            if pred.argmax() == label.argmax():
                cnt += 1

        result["f_measures"]["cluster"+str(i)] = {
            "trainData":len(x_train_a),
            "validData":len(x_val_a),
            "O":f_measure(predict,[0]),
            "T":f_measure(predict,[1]),
            "X":f_measure(predict,[2]),
            "T-X":f_measure(predict,[1,2]),
        }
        result["history"].append(history.history)
        result["modelpath"].append(namehead+str(i)+".weight")
        decoder.save_weights(namehead+str(i)+".weight")
    result["accuracy"] = cnt/total

    with open(namehead+".pickle",'wb') as f:
        pickle.dump(result,f)
    joblib.dump(knn_model,namehead+".knn")
    pprint(result["f_measures"])
    print("accuracy:",cnt/total)

