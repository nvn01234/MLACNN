from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Embedding, Flatten, Conv2D, \
    RepeatVector, Activation, Reshape, Multiply, Permute, Average
from keras.engine import Model, Layer
from keras import backend as K
import numpy as np
from keras.initializers import TruncatedNormal, Constant


def build_model():
    # input representation features
    words_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    # chars_input = Input(shape=[SEQUENCE_LEN, WORD_LEN], dtype='int32')
    pos1_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    pos2_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    tags_input = Input(shape=[SEQUENCE_LEN], dtype='int32')

    # lexical features
    e1_input = Input(shape=[ENTITY_LEN], dtype='int32')  # L1
    e2_input = Input(shape=[ENTITY_LEN], dtype='int32')  # L2
    e1context_input = Input(shape=[2], dtype='int32')  # L3
    e2context_input = Input(shape=[2], dtype='int32')  # L4

    # word embedding
    we = np.load("data/embedding/word_embeddings.npy")
    words_embed = Embedding(
        input_dim=we.shape[0],
        output_dim=we.shape[1],
        weights=[we],
        trainable=False,
    )
    words = words_embed(words_input)
    e1 = words_embed(e1_input)
    e2 = words_embed(e2_input)
    e1context = words_embed(e1context_input)
    e2context = words_embed(e2context_input)

    # lexical feature
    e1_flat = Flatten()(e1)
    e2_flat = Flatten()(e2)
    e1context_flat = Flatten()(e1context)
    e2context_flat = Flatten()(e2context)

    # position embedding
    pe1 = np.load("data/embedding/position_embeddings_1.npy")
    pos1 = Embedding(
        input_dim=pe1.shape[0],
        output_dim=pe1.shape[1],
        weights=[pe1],
    )(pos1_input)
    pe2 = np.load("data/embedding/position_embeddings_2.npy")
    pos2 = Embedding(
        input_dim=pe2.shape[0],
        output_dim=pe2.shape[1],
        weights=[pe2],
    )(pos2_input)

    # tag embedding
    te = np.load("data/embedding/tag_embeddings.npy")
    tags = Embedding(
        input_dim=te.shape[0],
        output_dim=te.shape[1],
        weights=[te],
    )(tags_input)

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
    input_repre = Concatenate()([words, pos1, pos2, tags])
    input_repre = Dropout(DROPOUT)(input_repre)

    # attention input
    # e1_repeat = RepeatVector(SEQUENCE_LEN)(e1_flat)
    # h1 = Concatenate()([words, e1_repeat])
    # e2_repeat = RepeatVector(SEQUENCE_LEN)(e2_flat)
    # h2 = Concatenate()([words, e2_repeat])
    # MLP1 = Dense(units=ATT_HIDDEN_LAYER, activation="tanh")
    # MLP2 = Dense(units=1, activation="softmax")
    # u1 = MLP1(h1)
    # alpha1 = MLP2(u1)
    # u2 = MLP1(h2)
    # alpha2 = MLP2(u2)
    # alpha = Average()([alpha1, alpha2])
    # alpha = Reshape([SEQUENCE_LEN])(alpha)
    # alpha = RepeatVector(WORD_REPRE_SIZE)(alpha)
    # alpha = Permute([2, 1])(alpha)
    # input_repre = Multiply()([input_repre, alpha])

    # word-level convolution
    pooled_word = []
    for size in WINDOW_SIZES_WORD:
        conv = Conv1D(filters=NB_FILTERS_WORD,
                      kernel_size=size,
                      padding="same",
                      activation="relu",
                      kernel_initializer=TruncatedNormal(stddev=0.1),
                      bias_initializer=Constant(0.1),
                      )(input_repre)
        pool = GlobalMaxPool1D()(conv)
        pooled_word.append(pool)

    # fully connected
    output = Concatenate()([*pooled_word, e1_flat, e2_flat, e1context_flat, e2context_flat])
    output = Dropout(DROPOUT)(output)
    output = Dense(
        units=NB_RELATIONS,
        activation="softmax",
        kernel_initializer=TruncatedNormal(stddev=0.1),
        bias_initializer=Constant(0.1),
        kernel_regularizer='l2',
        bias_regularizer='l2',
    )(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, tags_input, e1_input, e2_input, e1context_input, e2context_input], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer='adam')
    # model.summary()
    return model


class CharLevelPooling(Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[3]

    def call(self, inputs, **kwargs):
        return K.max(inputs, axis=2)


if __name__ == "__main__":
    build_model()
