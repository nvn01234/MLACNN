from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Embedding, Flatten, Conv2D
from keras.engine import Model, InputSpec, Layer
from keras import backend as K
import numpy as np
from keras.initializers import TruncatedNormal, Constant
from keras import initializers, regularizers


def build_model():
    # input
    words_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    # chars_input = Input(shape=[SEQUENCE_LEN, WORD_LEN], dtype='int32')
    pos1_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    pos2_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    e1_input = Input(shape=[3], dtype='int32')
    e2_input = Input(shape=[3], dtype='int32')

    # word embedding
    we = np.load("data/embedding/word_embeddings.npy")
    words_embed = Embedding(we.shape[0], we.shape[1], weights=[we], trainable=False)
    words = words_embed(words_input)
    e1 = words_embed(e1_input)
    e2 = words_embed(e2_input)

    # position embedding
    pos1 = Embedding(
        input_dim=NB_POSITIONS, output_dim=POSITION_EMBED_SIZE,
        embeddings_initializer=TruncatedNormal(stddev=0.1),
    )(pos1_input)
    pos2 = Embedding(
        input_dim=NB_POSITIONS, output_dim=POSITION_EMBED_SIZE,
        embeddings_initializer=TruncatedNormal(stddev=0.1),
    )(pos2_input)

    # character embedding
    # ce = np.load("data/embedding/char_embeddings.npy")
    # chars_embed = Embedding(ce.shape[0], ce.shape[1], weights=[ce], trainable=False)
    # chars = chars_embed(chars_input)

    # character-level convolution
    # pooled_char = []
    # for size in WINDOW_SIZES_CHAR:
    #     chars = Conv2D(filters=NB_FILTERS_CHAR,
    #                    kernel_size=(1, size),
    #                    padding="same",
    #                    activation="relu",
    #                    kernel_initializer=TruncatedNormal(stddev=0.1),
    #                    bias_initializer=Constant(0.1),
    #                    )(chars)
    #     pool = CharLevelPooling()(chars)
    #     pooled_char.append(pool)

    # input representation
    input_repre = Concatenate()([words, pos1, pos2])
    input_repre = Dropout(DROPOUT)(input_repre)

    # word-level convolution
    rel_embed = Embedding(
        input_dim=NB_RELATIONS, output_dim=NB_FILTERS_WORD,
        embeddings_initializer=TruncatedNormal(stddev=0.1),
    )
    rel_embed.build(None)
    rel_embed = rel_embed.weights
    pooled_word = []
    for size in WINDOW_SIZES_WORD:
        conv = Conv1D(filters=NB_FILTERS_WORD,
                      kernel_size=size,
                      padding="same",
                      activation="relu",
                      kernel_initializer=TruncatedNormal(stddev=0.1),
                      bias_initializer=Constant(0.1),
                      )(input_repre)
        pool = AttentionPooling(rel_embed)(conv)
        # pool = GlobalMaxPool1D()(conv)
        pooled_word.append(pool)

    # lexical feature
    e1_flat = Flatten()(e1)
    e2_flat = Flatten()(e2)

    # fully connected
    output = Concatenate()([*pooled_word, e1_flat, e2_flat])
    output = Dropout(DROPOUT)(output)
    output = Dense(
        units=NB_RELATIONS,
        activation="softmax",
        kernel_initializer=TruncatedNormal(stddev=0.1),
        bias_initializer=Constant(0.1),
        kernel_regularizer='l2',
        bias_regularizer='l2',
    )(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer='adam')
    # model.summary()
    return model


class AttentionPooling(Layer):
    def __init__(self, rel_embed, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.rel_embed = rel_embed

    def build(self, input_shape):
        self.U = self.add_weight(
            name="U",
            shape=[input_shape[-1], input_shape[-1]],
            initializer=TruncatedNormal(stddev=0.1),
            regularizer=regularizers.get(None)
        )
        self.built = True

    def call(self, inputs, **kwargs):
        G = K.dot(inputs, self.U)
        rel_embed = K.reshape(K.transpose(self.rel_embed), [NB_FILTERS_WORD, NB_RELATIONS])
        G = K.dot(G, rel_embed)
        AP = K.softmax(G)

        wo = K.batch_dot(inputs, AP, [1, 1])
        wo = K.max(wo, -1)
        return wo

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class CharLevelPooling(Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[3]

    def call(self, inputs, **kwargs):
        return K.max(inputs, axis=2)


if __name__ == "__main__":
    build_model()
