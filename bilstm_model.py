import keras
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense

import re
import sys
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import math
import numpy as np
from pprint import pprint


class BidirectionalLSTMModel:

    def __init__(self, maxlen, input_dim, output_dim, n_hidden):
        self.maxlen = maxlen
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

    def build_model(self):
        input_ = Input(shape=(self.maxlen, self.input_dim), dtype='float32')  # todo: maxlen->Noneでいいかも
        encoder = Bidirectional(LSTM(self.n_hidden, return_sequences=True))(input_)
        output_ = Dense(self.output_dim, activation='softmax')(encoder)

        model = Model(inputs=input_, outputs=output_)
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


def load_json_as_dict(json_file):
    with open(json_file, "r") as f:
        data_dict = json.load(f)
    return data_dict


def flatten_dict(data_dict):
    flattened_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                flattened_dict[k_] = v_
        else:
            flattened_dict[k] = v
    return flattened_dict


def load_race_data():
    score_race_lists = []

    current_dir = Path.cwd()
    results_fir = current_dir / 'results'
    json_files = results_fir.glob('2003*.json')

    effective_score_labels = [
        "arrival_order",
        "frame_no",
        "horse_no",
        # "horse_name",
        "horse_sex",
        "horse_age",
        "horse_weight",
        "horse_weight_diff",
        # "horse_b",
        # "arrival_diff",
        "time",
        "last3f_time",
        "passing_order_1st",
        "passing_order_2nd",
        "passing_order_3rd",
        "passing_order_4th",
        # "jockey_name",
        "jockey_weight",
        "odds",
        "popularity",
        # "trainer_name",
    ]

    effective_racehead_labels = [
        "race_no",
        # "date",
        "week",
        "kai",
        # "lacation",
        "nichi",
        "start_time",
        "weather",
        "condition",
    ]

    # jsonファイルを全読み込み
    for json_file in json_files:
        data_dict = load_json_as_dict(json_file)

        # ネスト化された辞書を平坦化
        score_dicts = []
        racehead_dict = {}
        for k, v in data_dict.items():
            if k == 'scores':
                for s in v:
                    flattened_dict = flatten_dict(s)
                    score_dict = {k_: v_ for k_, v_ in flattened_dict.items() if k_ in effective_score_labels}
                    score_dicts.append(score_dict)
            elif k == 'recehead':
                flattened_dict = flatten_dict(v)
                racehead_dict = {k_: v_ for k_, v_ in flattened_dict.items() if k_ in effective_racehead_labels}

        # データを整形
        score_race = []
        for score_dict in score_dicts:
            # score
            for k, v in score_dict.items():
                if k == 'horse_sex':
                    if v == '牡':
                        score_race.append(0)
                    elif v == '牝':
                        score_race.append(1)
                    elif v == 'せん':
                        score_race.append(2)
                elif k == 'jockey_weight':
                    if isinstance(v, str):
                        data = re.sub(re.compile("[☆△▲★◇]"), "", v)
                        score_race.append(float(data))
                    elif isinstance(v, float):
                        score_race.append(v)
                else:
                    score_race.append(v)
            # racehead
            for k, v in racehead_dict.items():
                if k == 'week':
                    if v == '日':
                        score_race.append(0)
                    elif v == '月':
                        score_race.append(1)
                    elif v == '火':
                        score_race.append(2)
                    elif v == '水':
                        score_race.append(3)
                    elif v == '木':
                        score_race.append(4)
                    elif v == '金':
                        score_race.append(5)
                    elif v == '土':
                        score_race.append(6)
                elif k == 'start_time':
                    base_time = datetime.strptime("00:00", "%H:%M")
                    time = datetime.strptime(v, "%H:%M")
                    min = abs(time - base_time).total_seconds / 60.0
                    score_race.append(min)
                elif k == 'weather':
                    if v == '晴':
                        score_race.append(0)
                    elif v == '曇':
                        score_race.append(1)
                    elif v == '雨':
                        score_race.append(2)
                    elif v == '雪':
                        score_race.append(3)
                elif k == 'condition':
                    if v == '良':
                        score_race.append(0)
                    elif v == '稍重':
                        score_race.append(1)
                    elif v == '重':
                        score_race.append(2)
                    elif v == '不良':
                        score_race.append(3)
                else:
                    score_race.append(v)
        score_race_lists.append(score_race)
        pprint(score_race)
    return score_race_lists


if __name__ == '__main__':
    score_race_lists = load_race_data()
