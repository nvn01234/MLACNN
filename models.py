from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Embedding, Flatten, Conv2D
from keras.engine import Model, InputSpec, Layer
from keras import backend as K
from keras.optimizers import Adam
import numpy as np


def build_model():
    # input
    words_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    chars_input = Input(shape=[SEQUENCE_LEN, WORD_LEN], dtype='int32')
    pos1_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    pos2_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    e1_input = Input(shape=[3], dtype='int32')
    e2_input = Input(shape=[3], dtype='int32')

    # word embedding
    # we = np.load("data/embedding/word_embeddings.npy")
    we = np.random.random([10, WORD_EMBED_SIZE])
    words_embed = Embedding(we.shape[0], we.shape[1], weights=[we], trainable=False)
    words = words_embed(words_input)
    e1 = words_embed(e1_input)
    e2 = words_embed(e2_input)

    # position embedding
    pos1 = Embedding(NB_POSITIONS, POSITION_EMBED_SIZE)(pos1_input)
    pos2 = Embedding(NB_POSITIONS, POSITION_EMBED_SIZE)(pos2_input)

    # character embedding
    # ce = np.load("data/embedding/char_embeddings.npy")
    ce = np.random.random([10, CHAR_EMBED_SIZE])
    chars_embed = Embedding(ce.shape[0], ce.shape[1], weights=[ce], trainable=False)
    chars = chars_embed(chars_input)

    # character-level convolution
    chars = Conv2D(filters=NB_FILTERS_CHAR,
                   kernel_size=(1, WINDOW_SIZE_CHAR),
                   padding="same",
                   activation="relu")(chars)
    chars = GlobalMaxPool1D4dim()(chars)

    # input representation
    input_repre = Concatenate()([words, chars, pos1, pos2])
    input_repre = Dropout(DROPOUT)(input_repre)

    # word-level convolution
    pooled = []
    for size in WINDOW_SIZES_WORD:
        conv = Conv1D(filters=NB_FILTERS_WORD,
                      kernel_size=size,
                      padding="same",
                      activation="relu")(input_repre)
        pool = GlobalMaxPool1D()(conv)
        pooled.append(pool)

    # lexical feature
    e1_flat = Flatten()(e1)
    e2_flat = Flatten()(e2)

    # fully connected
    output = Concatenate()([*pooled, e1_flat, e2_flat])
    output = Dropout(DROPOUT)(output)
    output = Dense(units=NB_RELATIONS, activation="softmax")(output)

    model = Model(inputs=[words_input, chars_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    optimizer = Adam(LEARNING_RATE)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    # model.summary()
    return model


class GlobalMaxPool1D4dim(Layer):
    def __init__(self, **kwargs):
        super(GlobalMaxPool1D4dim, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[3]

    def call(self, inputs, **kwargs):
        return K.max(inputs, axis=2)


if __name__ == "__main__":
    build_model()
