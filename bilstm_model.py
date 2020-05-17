import keras
from keras.preprocessing import sequence
from keras.models import Model
from keras.models import save_model, load_model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from keras.layers import TimeDistributed, Bidirectional
from keras.utils import plot_model

import re
import sys
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import math
import numpy as np
from pprint import pprint

from scraper import load_race_data, fetch_predicting_data


class BidirectionalLSTMModel:

    def __init__(self, maxlen, input_dim, output_dim, n_hidden):
        self.maxlen = maxlen
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

    def build_model(self):
        # Encoder
        # enc_input = Input(shape=(self.maxlen, self.input_dim), dtype='float32', name='encoder_input')  # todo: maxlen->Noneでいいかも
        # enc_bilstm, [state_h, state_c] = Bidirectional(LSTM(self.n_hidden, name='encoder_bilstm', return_sequences=True))(enc_input)
        # enc_state = [state_h, state_c]
        # enc_model = Model(input_shape=enc_input, outputs=[enc_bilstm, state_h, state_c])

        # Decoder
        # dec_lstm = LSTM(self.n_hidden, name='decoder_lstm', return_state=True)()
        # dec_dense = Dense(self.output_dim, activation='softmax', name='decoder_dense')

        inputs = Input(shape=(self.maxlen, self.input_dim), dtype='float32', name='inputs')
        bilstm = Bidirectional(LSTM(self.n_hidden, return_sequences=True, activation='tanh', name='bilstm1'),
                                                    input_shape=(self.maxlen, self.input_dim))(inputs)
        # state_h = Concatenate()([f_h, b_h])
        # state_c = Concatenate()([f_c, b_c])
        # bilstm = Bidirectional(LSTM(self.n_hidden, activation='tanh', name='bilstm2'))(state_h)
        dense = TimeDistributed(Dense(self.output_dim, activation='softmax', name='dense'),
                                input_shape=(self.maxlen, self.n_hidden))(bilstm)

        model = Model(inputs=inputs, outputs=dense)
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['categorical_accuracy'])
        return model

    def train(self, x_input, y_input, batch_size, epochs):
        print("[bilstm] build ...")
        model = self.build_model()

        x_i = x_input
        y_i = y_input
        z = list(zip(x_i, y_i))
        np.random.shuffle(z)  # シャッフル
        x_i, y_i = zip(*z)  # unzip z into x_i and y_i

        x_i = np.array(x_i).reshape(len(x_i), self.maxlen)
        y_i = np.array(y_i).reshape(len(y_i), self.maxlen)

        n_split = int(x_i.shape[0] * 0.9)  # 訓練データとテストデータを9:1に分割
        x_train, x_val = np.vsplit(x_i, [n_split])
        y_train, y_val = np.vsplit(y_i, [n_split])

        print("[bilstm] train ...")
        for j in range(0, epochs):
            print("Epoch {}/{}".format(j + 1, epochs))
            self.on_batch(model, x_train, y_train, x_val, y_val)
        return model

    def on_batch(self, model, x_train, y_train, x_val, y_val, batch_size):
        # 損失関数、評価関数の平均計算用リスト
        list_loss = []
        list_accuracy = []

        s_time = time.time()
        row = x_train.shape[0]  # 1バッチに含まれるサンプル数
        n_batch = math.ceil(row / batch_size)  # x_trainに含まれるバッチ数

        for i in range(0, n_batch):
            s = i * batch_size
            e = min([(i + 1) * batch_size, row])  # x_trainのケツインデックスを考慮している
            x_on_batch = x_train[s: e, :]
            y_on_batch = y_train[s: e, :]
            result = model.train_on_batch(x_on_batch, y_on_batch)
            list_loss.append(result[0])  # todo: モデルが単一の出力かつ評価関数なし？なので戻り値はスカラ値の可能性あり
            list_accuracy.append(result[1])

            elased_time = time.time() - s_time
            sys.stdout.write("TRAIN\r{}/{} time: {}s\tloss: {0:.4f}\taccuracy: {0:.4f}".format(
                str(e), str(row), str(int(elased_time)),
                np.average(list_loss), np.average(list_accuracy)))
            sys.stdout.flush()
            del x_on_batch, y_on_batch

        self.valid(model, x_val, y_val, batch_size)
        return

    def valid(self, model, x_val, y_val, batch_size):
        list_loss = []
        list_accuracy = []
        row = x_val.shape[0]

        s_time = time.time()
        n_batch = math.ceil(row / batch_size)

        n_loss = 0
        sum_loss = 0.0
        for i in range(0, n_batch):
            s = i * batch_size
            e = min([(i + 1) + batch_size, row])
            x_on_batch = x_val[s: e, :]
            y_on_batch = y_val[s: e, :]
            result = model.test_on_batch(x_on_batch, y_on_batch)
            list_loss.append(result[0])
            list_accuracy.append(result[1])

            elased_time = time.time() - s_time
            sys.stdout.write("VALID\r{}/{} time: {}s\tloss: {0:.4f}\taccuracy: {0:.4f}".format(
                str(e), str(row), str(int(elased_time)),
                np.average(list_loss), np.average(list_accuracy)))
            sys.stdout.flush()
            del x_on_batch, y_on_batch
        return


    def get_batch(self):
        pass


# def arrival_order_to_one_hot(arrival_order, maxlen):
    # one_hot = np.identity(maxlen)arrival_order
    # one_hot = np.zeros((maxlen, maxlen))
    # for i in arrival_order:
    #     one_hot[i][i] = 1.0
    # return one_hot


def load(maxlen):
    list_score_and_racehead, list_arrival_order = load_race_data()
    # labels = arrival_order_to_one_hot(list_arrival_order, 18)
    list_label = []
    for arrival_order in list_arrival_order:
        try:
            label = np.eye(maxlen)[arrival_order]
        except IndexError as e:
            print(e)
        if label.shape[0] < maxlen:
            for i in range(maxlen - label.shape[0]):
                zero_label = np.zeros(maxlen)
                label = np.vstack((label, zero_label))
        list_label.append(label)

    train_x = np.array(list_score_and_racehead)
    train_y = np.array(list_label)

    print(train_x.shape)
    print(train_y.shape)

    return train_x, train_y

def train(train_x, train_y, maxlen, model_name):
    # Hyper parameters
    input_dim = 17
    output_dim = 18
    n_hidden = 256
    epochs = 100
    batch_size = 32

    BLM = BidirectionalLSTMModel(maxlen=maxlen,
                                 input_dim=input_dim,
                                 output_dim=output_dim,
                                 n_hidden=n_hidden)
    model = BLM.build_model()
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1)

    plot_model(model, show_shapes=True, to_file='{}.png'.format(model_name))
    save_model('{}.h5'.format(model_name))

    return model


def predict(model_name, input_dim):
    model = load_model('{}.h5'.format(model_name))
    url = 'https://keiba.yahoo.co.jp/race/result/2005020811/'

    score_and_racehead = fetch_predicting_data(url)

    # popularity
    odds = []
    for sr in score_and_racehead:
        odds.append(sr[9])
    popurarities = np.array(odds).argsort()
    popurs = []
    for i in range(maxlen):
        if i >= popurarities.shape[0]:
            break
        #     popurarities = np.append(popurarities, 0)
        # else:
        popurs.append([popurarities[i] + 1])
    popurs = np.array([popurs])

    test_x = np.array([score_and_racehead])
    test_x = np.insert(test_x, [10], popurs, axis=2)

    if test_x.shape[0] < maxlen:
        for i in range(maxlen - test_x.shape[1]):
            zero_label = np.array([[[0] * input_dim]])  # input_dim = 17
            test_x = np.append(test_x, zero_label, axis=1)

    y = model.predict(test_x, batch_size=1, verbose=0, steps=None)
    return y


if __name__ == '__main__':
    maxlen = 18

    # Load
    train_x, train_y = load(maxlen)

    model_name = 'bilstm_model'

    # Train
    model = train(train_x, train_y, maxlen, model_name)

    # Infer
    # y = predict(model_name, input_dim=17)
    # pprint(y)


