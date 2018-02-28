import numpy as np
import tensorflow as tf
from keras.layers.pooling import _GlobalPooling1D
from tensorflow import ConfigProto
from keras.layers import Layer, Lambda, Activation, Add, Multiply, Concatenate, Conv1D, GlobalMaxPool1D, Embedding, \
    Input, Reshape, RepeatVector, Dot, Dense
from keras import backend as K, Model
from keras.engine import InputSpec
from settings import *


def build_model():
    print("load word_embeddings, position_embeddings")
    we = np.load("data/word_embeddings.npy")

    words_input = Input(shape=(SEQUENCE_LEN,), name="words_input")
    pos1_input = Input(shape=(SEQUENCE_LEN,), name="pos1_input")
    pos2_input = Input(shape=(SEQUENCE_LEN,), name="pos2_input")
    e1_input = Input(shape=(), name="e1_input")
    e2_input = Input(shape=(), name="e2_input")

    word_embeddings = Embedding(
        input_dim=we.shape[0],
        output_dim=we.shape[1],
        trainable=False,
        name="word_embeddings",
    )
    position_embeddings = Embedding(
        input_dim=NB_DISTANCES,
        output_dim=POSITION_EMBED_SIZE,
        name="position_embeddings",
    )


    wd = word_embeddings(words_input)
    wp1 = position_embeddings(pos1_input)
    wp2 = position_embeddings(pos2_input)
    e1 = word_embeddings(e1_input)
    e2 = word_embeddings(e2_input)
    wM = Concatenate(name="word_representation")([wd, wp1, wp2])

    # Attention input
    r = AttentionInput(name="attention_input")([wM, wd, e1, e2])

    # Convolution
    R_star = Conv1D(
        kernel_size=WINDOW_SIZE,
        filters=NB_FILTERS,
        activation="tanh",
        name="convolution",
    )(r) # (?, SEQUENCE_LEN, NB_FILTERS)

    wo = AttentionPooling()(R_star)
    output = Dense(NB_RELATIONS)(wo)
    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentroy', metrics=['accuracy'], optimizer='sgd')
    model.summary()
    return model



class AttentionInput(Layer):
    def __init__(self, **kwargs):
        super(AttentionInput, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=2), InputSpec(ndim=2)]

    def call(self, inputs, **kwargs):
        wM, wd, e1, e2 = inputs

        e1 = K.repeat(e1, SEQUENCE_LEN)  # (?, SEQUENCE_LEN, WORD_EMBED_SIZE)
        A1 = K.batch_dot(wd, e1, -1)
        A1 = K.reshape(A1, (-1, SEQUENCE_LEN))
        alpha1 = K.softmax(A1)

        e2 = K.repeat(e2, SEQUENCE_LEN)  # (?, SEQUENCE_LEN, WORD_EMBED_SIZE)
        A2 = K.batch_dot(wd, e2, -1)
        A2 = K.reshape(A2, (-1, SEQUENCE_LEN))
        alpha2 = K.softmax(A2)

        alpha = (alpha1 ** 2 + alpha2 ** 2)/2 # (?, SEQUENCE_LEN)
        r = wM * K.expand_dims(alpha)
        print(K.shape(r))
        return r

class AttentionPooling(_GlobalPooling1D):
    def build(self, input_shape):
        self.U = self.add_weight(
            name="U",
            shape=(NB_FILTERS, NB_RELATIONS),
        )
        self.WL = self.add_weight(
            name="WL",
            shape=(NB_RELATIONS, NB_FILTERS),
        )
        self.built = True

    def call(self, R_star, **kwargs):
        G = K.dot(R_star, self.U) # (?, SEQUENCE_LEN, NB_RELATIONS)
        G = K.dot(G, self.WL)  # (?, SEQUENCE_LEN, NB_FILTERS)
        AP = K.softmax(G)
        wo = K.batch_dot(R_star, AP, (1, 2)) #(?, NB_FILTERS, NB_FILTERS)
        wo = K.max(wo, -1) # (?, NB_FILTERS)
        return wo

