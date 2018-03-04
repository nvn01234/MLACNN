from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Reshape, Layer, RepeatVector
from keras.engine import Model
from keras import backend as K
from keras import  initializers, regularizers

def build_model():
    words_input = Input(shape=[SEQUENCE_LEN, WORD_EMBED_SIZE], dtype='float32')
    pos1_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], dtype='float32')
    pos2_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], dtype='float32')
    e1_input = Input(shape=[WORD_EMBED_SIZE], dtype='float32')
    e2_input = Input(shape=[WORD_EMBED_SIZE], dtype='float32')

    input_repre = Concatenate()([words_input, pos1_input, pos2_input])
    input_repre = Dropout(DROPOUT)(input_repre)
    input_att = InputAttention()([input_repre, words_input, e1_input, e2_input])

    conv = Conv1D(filters=NB_FILTERS,
                  kernel_size=WINDOW_SIZE,
                  padding="same",
                  activation="tanh")(input_att)
    pool = GlobalMaxPool1D()(conv)
    output = Dropout(DROPOUT)(pool)
    output = Dense(units=NB_RELATIONS, activation="softmax")(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer='adam')
    model.summary()
    return model


class InputAttention(Layer):
    def compute_output_shape(self, input_shape):
        return [input_shape[0], SEQUENCE_LEN, WORD_REPRE_SIZE]

    def build(self, input_shape):
        self.W = self.add_weight([WORD_EMBED_SIZE, WORD_EMBED_SIZE],
                                 initializer=initializers.get('truncated_normal'),
                                 regularizer=regularizers.get(None),
                                 name="W")
        self.built = True

    def call(self, inputs, **kwargs):
        input_repre, words_input, e1, e2 = inputs

        A1 = K.dot(words_input, self.W)  # (?, SEQUENCE_LEN, WORD_EMBED_SIZE)
        A1 = K.batch_dot(A1, e1, (2, 1))  # (?, SEQUENCE_LEN, 1)
        alpha1 = K.softmax(A1)

        A2 = K.dot(words_input, self.W)
        A2 = K.batch_dot(A2, e2, (2, 1))
        alpha2 = K.softmax(A2)

        alpha = (alpha1 + alpha2)/2
        alpha = K.expand_dims(alpha)
        output = input_repre * alpha
        return output


if __name__ == "__main__":
    model = build_model()