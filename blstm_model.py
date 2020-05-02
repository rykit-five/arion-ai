import keras
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense


class BidirectionalLSTMModel:

    def __init__(self, maxlen, input_dim, output_dim, n_hidden):
        self.maxlen = maxlen
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

    def build_model(self):
        print("[model] build ...")
        input_ = Input(shape=(self.maxlen, self.input_dim), dtype='float32')
        encoder = Bidirectional(LSTM(self.n_hidden, return_sequences=True))(input_)
        output_ = Dense(self.output_dim, activation='softmax')(encoder)

        model = Model(inputs=input_, outputs=output_)
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['categorical_accuracy'])
        return model

    def train(self, model, x_input, y_input, batch_size, epoch):

        x_train, y_train = None
        x_val, y_val = None

        print("[model] train ...")
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=[x_val, y_val])

    def get_batch(self):
        pass
