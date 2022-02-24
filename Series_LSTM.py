
# coding: utf-8

# In[1]:


import pandas as pd

# In[2]:


# 参考
# https://github.com/kenchin110100/machine_learning/blob/master/sampleSeq2Sep.py
# http://qiita.com/kenchin110100/items/b34f5106d5a211f4c004
from chainer import functions, optimizers
from chainer import Chain, Variable, functions, cuda, links, optimizer, optimizers, serializers
import numpy as np
import cupy as cp
from preprocess_chainer import mk_minibatch
from os import system
# xp = np
xp = cp

# In[7]:


class LSTM_usr(Chain):
    def __init__(self, hidden_size, emb_size):
        super(LSTM_usr, self).__init__(
            # # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            # eh = links.Linear(emb_size, hidden_size),
            # # 出力された中間層を4倍のサイズに変換するための層
            # hh = links.Linear(hidden_size, 4 * hidden_size)
            lstm=links.LSTM(emb_size, hidden_size)
        )

    def __call__(self, e):
        """
        Encoderの動作
        :param e: 埋め込みベクトル
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        return self.lstm(e)
    
    def r(self):
        self.lstm.reset_state()

# In[8]:


class LSTM_sys(Chain):
    def __init__(self, hidden_size, emb_size, dr_ratio=0.5):
        """
        クラスの初期化
        :param hidden_size: 中間ベクトルのサイズ
        """
        super(LSTM_sys, self).__init__(
            # # 単語ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
            # eh=links.Linear(emb_size, 4 * hidden_size),
            # # 中間ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
            # hh=links.Linear(hidden_size, 4 * hidden_size),
            # # 出力されたベクトルをラベルのサイズに変換する層
            # he=links.Linear(hidden_size, 3)
            lstm=links.LSTM(emb_size, hidden_size),
            ot=links.Linear(hidden_size, 3)
        )
        self.ratio = dr_ratio

    def __call__(self, e):
        """
        :param e: 埋め込みベクトル
        :param c: 内部メモリ
        :param h: 中間ベクトル
        :return: 予測単語、次の内部メモリ、次の中間ベクトル
        """
        # # 内部メモリ、単語ベクトルの4倍+中間ベクトルの4倍をLSTMにかける
        # c, h = functions.lstm(c, self.eh(e) + self.hh(h))
        # # 出力された中間ベクトルをラベルに、ラベルを確率密度に変換
        # t = functions.softmax(self.he(h))
        x = self.ot(functions.dropout(functions.tanh(self.lstm(e)), self.ratio))
        # print("x: ", x)
        return functions.softmax(x)
    
    def r(self):
        self.lstm.reset_state()
    
    def s(self, h):
        self.lstm.h = h


class Series_LSTM(Chain):
    def __init__(self, emb_size=512, hidden_size=512, batch_size=50, parameters=None, dr_ratio_sys=0.5):
        """
        Seq2Seqの初期化
        :param hidden_size: 中間ベクトルのサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dr_ratio_sys = dr_ratio_sys
        if parameters is not None:
            if "hidden_size" in parameters.keys():
                self.hidden_size = parameters["hidden_size"]
            if "batch_size" in parameters.keys():
                self.batch_size = parameters["batch_size"]
            if "dr_ratio_sys" in parameters.keys():
                self.dr_ratio_sys = parameters["dr_ratio_sys"]
        super(Series_LSTM, self).__init__(
            # Encoderのインスタンス化
            encoder=LSTM_usr(hidden_size, emb_size),
            # Decoderのインスタンス化
            decoder=LSTM_sys(hidden_size, emb_size, self.dr_ratio_sys)
        )

    def usr(self, embedding):
        """
        Encoderを計算する部分
        :param words: 単語が記録されたリスト
        :return:
        """

        # エンコーダーに単語を順番に読み込ませる
        for e in embedding:
            h = self.encoder(e)
            # print("self.h: ", self.encoder(e))
        return h
        # # 内部メモリは引き継がないので、初期化
        # self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    def sys(self, embedding, h):
        """
        デコーダーを計算する部分
        :param w: 単語
        :return: 単語数サイズのベクトルを出力する
        """
        self.decoder.s(h)
        for e in embedding:
            # print("self.h: ", self.decoder(e))
            t = self.decoder(e)
            
        # 内部メモリは引き継がないので、初期化
        #self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        return t

    def reset(self):
        """
        中間ベクトル、内部メモリ、勾配の初期化
        :return:
        """
        # self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # self.c = None
        # self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # self.h = None
        
        self.encoder.r()
        self.decoder.r()

    def __call__(self, usr_words, sys_words):
        self.reset()
        # usr_words = [Variable(xp.array(row)) for row in usr_words]
        usr_words = Variable(xp.asarray(usr_words, dtype=cp.float32))
        h = self.usr(usr_words)
        # sys_words = [Variable(xp.array(row)) for row in sys_words]
        sys_words = Variable(xp.asarray(sys_words, dtype=cp.float32))
        t = self.sys(sys_words, h)
        return t
    
    def forward(self, usr_words, sys_words, label, save_file=None, epoch=None):
        """
        順伝播の計算を行う関数
        :param enc_words: 発話文の単語を記録したリスト
        :param dec_words: 応答文の単語を記録したリスト
        :param model: Seq2Seqのインスタンス
        :param ARR: cuda.cupyかnumpyか
        :return: 計算した損失の合計
        """
        # バッチサイズを記録
        #batch_size = len(usr_words[0])
        # model内に保存されている勾配をリセット
        self.reset()
        # 発話リスト内の単語を、chainerの型であるVariable型に変更
        # usr_words = [Variable(xp.array(row)) for row in usr_words]
        usr_words = Variable(xp.asarray(usr_words, dtype=xp.float32))
        # エンコードの計算 ⑴
        h = self.usr(usr_words)
        # デコーダーの計算
        # sys_words = [Variable(xp.array(row)) for row in sys_words]
        sys_words = Variable(xp.asarray(sys_words, dtype=cp.float32))
        t = self.sys(sys_words, h)
        # print('t: ', t.data)
        label = Variable(xp.asarray(label, dtype=cp.float32))
        loss = functions.mean_squared_error(t, label)
        # print("loss: ", loss)
        if save_file is not None:
            system("echo %d,%.4f >> %s" % (epoch, cuda.to_cpu(loss.data), save_file))
        return loss

# In[ ]:


def train(emb_size, usr_words, sys_words, labels, max_epoch=50, model_name='models/Series_LSTM',
          hidden_size=1024, batch_size=11):
    # モデルのインスタンス化
    model = Series_LSTM(emb_size, hidden_size=hidden_size,
                    batch_size=batch_size)
    cuda.get_device_from_id(0).use()
    model.to_gpu()
    idx = np.arange(len(usr_words))
    
    # モデルの初期化
    model.reset()
    # エポックごとにoptimizerの初期化
    # 無難にAdamを使います
    opt = optimizers.Adam()
    # モデルをoptimizerにセット
    opt.setup(model)
    # 勾配が大きすぎる場合に調整する
    opt.add_hook(optimizer.WeightDecay())
    opt.add_hook(optimizer.Lasso(0.003))

    # 学習開始
    for epoch in range(max_epoch):
        opt.new_epoch()
        np.random.shuffle(idx)

        # バッチ学習のスタート
        for num in range(len(usr_words)//batch_size):
            # input('>')
            # 任意のサイズのミニバッチを作成
            usr_minibatch = mk_minibatch(usr_words[idx[num*batch_size: (num+1)*batch_size]], emb_size)
            sys_minibatch = mk_minibatch(sys_words[idx[num*batch_size: (num+1)*batch_size]], emb_size)
            print('usr_minibatch_shape: ', usr_minibatch.data.shape)
            print('sys_minibatch_shape: ', sys_minibatch.data.shape)
            label_minibatch = labels[idx[num*batch_size: (num+1)*batch_size]]
            # 順伝播で損失の計算
            # total_loss = model.forward(usr_words=usr_minibatch,
            #                            sys_words=sys_minibatch,
            #                            label=label_minibatch)
            # 誤差逆伝播で勾配の計算
            # print('loss: ', total_loss)
            # opt.use_cleargrads()
            # total_loss.backward()
            # 計算した勾配を使ってネットワークを更新
            opt.update(model.forward, usr_minibatch, sys_minibatch, label_minibatch)
            # 記録された勾配を初期化する
        # エポックごとにモデルの保存
        outputpath = model_name+'e%d.model' % epoch
        serializers.save_hdf5(outputpath, model)


# In[ ]:
if __name__ == '__main__':
    df_vec = pd.read_pickle('data/vecs.pickle')

    sys_X_train = df_vec['system'].values
    usr_X_train = df_vec['user'].values
    emb_size = max(e.shape[-1] for e in sys_X_train)
    # print('emb_size: ', emb_size)

    df_annotations = pd.read_pickle('./data/annotations.pickle')
    df_annotations = df_annotations.reset_index(drop=True)
    tmp = df_annotations.fillna(0).sum(axis=1).values

    label = xp.array([list(i) for i in tmp]) / xp.array([[xp.sum(x)]*3 for x in tmp])

    print('loading was done')

    train(emb_size, usr_X_train, sys_X_train, label, max_epoch=30, batch_size=50, hidden_size=512)
