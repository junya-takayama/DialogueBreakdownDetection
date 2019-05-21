from keras.models import Sequential, Model
import keras
import os
from keras.layers import Merge, LSTM, Dense,GRU, SimpleRNN, core, Dropout, InputLayer,Input
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Flatten, Lambda, TimeDistributed, Activation, Permute, RepeatVector
from keras.optimizers import Adam, SGD, Adagrad,Adamax, Nadam
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate, Average, Multiply, Add
from keras import backend as K
import numpy as np
import pandas as pd
import glob
import sys
from sklearn.cluster import KMeans
import utils
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=False
    )
)
set_session(tf.Session(config=config))

wvdim=512

def gen_conv(kernel_sizes,batch_input_shape):
    inp = Input(shape=batch_input_shape[1:])
    convs = []
    for kernel_size in kernel_sizes:
        conv1 = Conv1D(
                int(batch_input_shape[2]/2),
                kernel_size,
                padding='causal',
                activation='relu',
                strides=1,
                batch_input_shape=batch_input_shape,init='uniform',kernel_regularizer=l2(0.00005))(inp)
        pool = GlobalAveragePooling1D()(conv1)
        convs.append(pool)

    if len(kernel_sizes) > 1:
        out = Concatenate()(convs)
    else:
        out = convs[0]

    return Model(input=inp, output=out)


def lstm_attn(kernel_sizes,batch_input_shape):
    inp = Input(shape=batch_input_shape[1:])
    unitsize = batch_input_shape[2]
    gru = LSTM(output_dim=unitsize, return_sequences=True,dropout_U=0.4,batch_input_shape=batch_input_shape)(inp)
    
    attention = TimeDistributed(Dense(1, activation='tanh'))(gru) 
    attention = Flatten()(attention)
    
    attention = Activation('softmax')(attention)
    attention = RepeatVector(unitsize)(attention)
    attention = Permute([2, 1])(attention)
    
    out = Multiply()([gru, attention])
    out = Lambda(lambda xin: K.sum(xin, axis=-2))(out)
    #out = Flatten()(out)

    return Model(input=inp, output=out)

def conv(kernel_sizes,batch_input_shape):
    inp = Input(shape=batch_input_shape[1:])
    filtersize = int(batch_input_shape[2]/2)
    convs = []
    for kernel_size in kernel_sizes:
        conv1 = inp
        for dilation_rate in [2 ** i for i in range(1)]:
            conv1 = Conv1D(
                    filtersize,
                    kernel_size,
                    padding='causal',
                    activation='relu',
                    dilation_rate = dilation_rate,
                    strides=1,
                    batch_input_shape=batch_input_shape,init='uniform',kernel_regularizer=l2(0.00005))(conv1)
        convs.append(conv1)
        
    if len(kernel_sizes) > 1:
        concat = Concatenate()(convs)
    else:
        concat = convs[0]

    out = GlobalAveragePooling1D()(concat)
    
    return Model(input=inp, output=out)
    
def conv_attn(kernel_sizes,batch_input_shape):
    inp = Input(shape=batch_input_shape[1:])
    filtersize = int(batch_input_shape[2]/2)
    convs = []
    for kernel_size in kernel_sizes:
        conv1 = Conv1D(
            filtersize,
            kernel_size,
            padding='causal',
            activation='relu',
            dilation_rate = 1,
            strides=1,
            batch_input_shape=batch_input_shape,init='uniform',kernel_regularizer=l2(0.00005))(inp)
        
        unitsize = filtersize
        attention = TimeDistributed(Dense(1, activation='relu'))(conv1) 
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(unitsize)(attention)
        attention = Permute([2, 1])(attention)
        attention = Multiply()([conv1, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=-2))(attention)
        convs.append(attention)
        
    if len(kernel_sizes) > 1:
        out = Concatenate()(convs)
    else:
        out = convs[0]
    unitsize = len(kernel_sizes) * filtersize
    #out = Flatten()(out)
    
    return Model(input=inp, output=out)

def conv_attn_2(kernel_sizes,batch_input_shape):
    inp = Input(shape=batch_input_shape[1:])
    filtersize = int(batch_input_shape[2]/2)
    convs = []
    for kernel_size in kernel_sizes:
        conv1 = Conv1D(
            filtersize,
            kernel_size,
            padding='causal',
            activation='relu',
            dilation_rate = 1,
            strides=1,
            batch_input_shape=batch_input_shape,init='uniform',kernel_regularizer=l2(0.00005)
        )(inp)
        conv2 = Conv1D(
            filtersize,
            kernel_size,
            padding='causal',
            activation='relu',
            dilation_rate = 1,
            strides=1,
            batch_input_shape=batch_input_shape,init='uniform',kernel_regularizer=l2(0.00005)
        )(inp)
        
        unitsize = filtersize
        attention = TimeDistributed(Dense(1, activation='relu'))(conv2) 
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(unitsize)(attention)
        attention = Permute([2, 1])(attention)
        attention = Multiply()([conv1, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=-2))(attention)
        convs.append(attention)
        
    if len(kernel_sizes) > 1:
        out = Concatenate()(convs)
    else:
        out = convs[0]
    unitsize = len(kernel_sizes) * filtersize
    #out = Flatten()(out)
    
    return Model(input=inp, output=out)



def direct_attn(kernel_sizes,batch_input_shape):
    inp = Input(shape=batch_input_shape[1:])
    unitsize = batch_input_shape[2]
    attention = TimeDistributed(Dense(1, activation='relu'))(inp) 
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(unitsize)(attention)
    attention = Permute([2, 1])(attention)
    
    out = Multiply()([inp, attention])
    out = Lambda(lambda xin: K.sum(xin, axis=-2))(out)
    out = Dense(256,activation='relu')(out)
    #out = Flatten()(out)
    
    return Model(input=inp, output=out)


class Classifier:
    def __init__(self,n_class=3,batch_size=100,pad_size=40,wvdim=512,weight=np.array([1,1,1]),kernel_sizes=[3,5]):
        encoder_a = Sequential()
        encoder_a.add(conv_attn_2(kernel_sizes,(None, pad_size, wvdim)))
        #encoder_a.add(GRU(output_dim=wvdim, return_sequences=True,dropout_U=0.6,batch_input_shape=(None, pad_size, wvdim)))
        
        encoder_b = Sequential()
        encoder_b.add(conv_attn_2(kernel_sizes,(None, pad_size, wvdim)))     
        #encoder_b.add(GRU(output_dim=wvdim, return_sequences=True,dropout_U=0.6,batch_input_shape=(None, pad_size, wvdim)))
        
        decoder = Sequential()
        decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
        #decoder.add(core.Dropout(0.3))
        #decoder.add(Dense(1000, activation='relu',kernel_regularizer=l2(0.001),init='uniform'))
        #decoder.add(BatchNormalization())
        decoder.add(core.Dropout(0.3))

        #decoder.add(Dense(512, activation='relu',kernel_regularizer=l2(0.00005),init='uniform'))
        decoder.add(Dense(512, activation='relu',kernel_regularizer=l2(0.00005),init='uniform'))
        #decoder.add(Dense(256, activation='relu',kernel_regularizer=l2(0.00005),init='uniform'))
        
        #decoder.add(BatchNormalization())
        #decoder.add(core.Dropout(0.3))
        decoder.add(Dense(n_class, activation='softmax'))
        adam = Adam(lr=0.001)
        adagrad = Adagrad()
        adamax = Adamax()
        nadam = Nadam(lr=0.001)
        sgd = SGD()
        #loss = Weighted(weight=weight).wpoisson
        loss = Weighted([1,1,1]).wmse
        decoder.compile(loss= loss,
                        optimizer=adam,
                       )
        
        self.decoder = decoder

class Weighted:
    def __init__(self,weight):
        self.weight = weight / np.sum(weight)
    def wmse(self,y_true, y_pred):
        return K.sum(K.square((y_pred  - y_true) * self.weight),axis=-1)
    def wmae(self,y_true, y_pred):
        return K.sum(K.abs(y_pred - y_true) * self.weight, axis=-1)
    def wkld(self,y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        return K.sum((y_true * K.log(y_true / y_pred)) * self.weight, axis=-1)
    def wjsd(self,y_true, y_pred):
        return 0.5 * self.wkld(y_true,y_pred) + 0.5 * self.wkld(y_pred,y_true)
    def wpoisson(self,y_true, y_pred):
        return K.mean((y_pred - y_true * K.log(y_pred + K.epsilon())) * self.weight, axis=-1)
    def wcos(self,y_true, y_pred):
        y_true = K.l2_normalize(y_true, axis=-1)
        y_pred = K.l2_normalize(y_pred, axis=-1)
        return -K.sum((y_true * y_pred) * self.weight, axis=-1)

class Extended:
    def __init__(self):
        self.mean = Weighted(np.array([1,1,1]))
    def jsd(self,y_true, y_pred):
        return self.mean.wjsd(y_true,y_pred)
    
def swish(x):
    return x * K.sigmoid(x)

if __name__ == "__main__":
    n_clusters = int(sys.argv[1])
    namehead = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) >= 4 else "kmeans" #あとrandomとall
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
    if mode == "kmeans":
        clusters = kmeans.fit_predict(kmeans_feature)
        annt_clusters = [list(map(lambda x:x[1],filter(lambda x:x[0] == i, zip(clusters,annotators)))) for i in range(n_clusters)]
    elif mode == "random":
        clusters = np.random.randint(0,n_clusters,len(kmeans_feature))
        annt_clusters = [list(map(lambda x:x[1],filter(lambda x:x[0] == i, zip(clusters,annotators)))) for i in range(n_clusters)]
    elif mode == "all":
        clusters = np.array([-1 for i in range(len(kmeans_feature))])
        annt_clusters = [annotators for i in range(n_clusters)]
    pad_size = 40
    
    
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
        decoder = Classifier(pad_size=pad_size,wvdim=wvdim,weight=inverseratio).decoder
        open('./models/classifier/'+namehead+'_k_'+str(n_clusters)+'.yaml','w').write(decoder.to_yaml())
        mc_cb = keras.callbacks.ModelCheckpoint(
            './models/classifier/'+namehead+'_k_'+str(n_clusters)+'_'+str(i)+'.weight', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True,
            save_weights_only=True, 
            mode='auto', 
            period=1
        )

        es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
        rl_cb = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.3, 
            patience=2, 
            verbose=1, 
            mode='auto', 
            epsilon=0.0005, 
            cooldown=2, 
            min_lr=0
        )

        history = decoder.fit(
            [user_x, system_x], 
            labels,
            batch_size=50, 
            nb_epoch=50,
            callbacks=[mc_cb,rl_cb,es_cb],
            validation_split = 0.15,
            shuffle=True,
        )
