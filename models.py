from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Embedding, Flatten, Conv2D, \
    RepeatVector, Activation, Reshape, Multiply, Permute, Average, Dot, Lambda
from keras.engine import Model, Layer
from keras import backend as K
import numpy as np
from keras.initializers import TruncatedNormal, Constant

from utils import make_dict


def build_model(embeddings):
    # input representation features
    words_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    chars_input = Input(shape=[SEQUENCE_LEN, WORD_LEN], dtype='int32')
    pos1_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    pos2_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    tags_input = Input(shape=[SEQUENCE_LEN], dtype='int32')

    # lexical features
    e1_input = Input(shape=[ENTITY_LEN], dtype='int32')  # L1
    e2_input = Input(shape=[ENTITY_LEN], dtype='int32')  # L2
    e1context_input = Input(shape=[2], dtype='int32')  # L3
    e2context_input = Input(shape=[2], dtype='int32')  # L4

    # word embedding
    we = embeddings["word_embeddings"]
    words_embed = Embedding(we.shape[0], we.shape[1], weights=[we], trainable=False)
    words = words_embed(words_input)
    e1 = words_embed(e1_input)
    e2 = words_embed(e2_input)
    e1context = words_embed(e1context_input)
    e2context = words_embed(e2context_input)

    # lexical feature
    word_conv = Conv1D(filters=WORD_EMBED_SIZE,
                       kernel_size=WINDOW_SIZE_WORD,
                       padding="same",
                       activation="relu",
                       kernel_initializer=TruncatedNormal(stddev=0.1),
                       bias_initializer=Constant(0.1))
    e1_conved = word_conv(e1)
    e1_pooled = GlobalMaxPool1D()(e1_conved)
    e2_conved = word_conv(e2)
    e2_pooled = GlobalMaxPool1D()(e2_conved)
    e1context = EntityContext()([e1_pooled, e1context])
    e2context = EntityContext()([e2_pooled, e2context])


    # position embedding
    pe1 = embeddings["position_embeddings_1"]
    pos1 = Embedding(pe1.shape[0], pe1.shape[1], weights=[pe1])(pos1_input)
    pe2 = embeddings["position_embeddings_2"]
    pos2 = Embedding(pe2.shape[0], pe2.shape[1], weights=[pe2])(pos2_input)

    # tag embedding
    te = embeddings["tag_embeddings"]
    tags = Embedding(te.shape[0], te.shape[1], weights=[te])(tags_input)

    # character embedding
    ce = embeddings["char_embeddings"]
    chars_embed = Embedding(ce.shape[0], ce.shape[1], weights=[ce], trainable=False)
    chars = chars_embed(chars_input)

    # character-level convolution
    char_conv = Conv2D(filters=NB_FILTERS_CHAR,
                       kernel_size=(1, WINDOW_SIZE_CHAR),
                       padding="same",
                       activation="relu",
                       kernel_initializer=TruncatedNormal(stddev=0.1),
                       bias_initializer=Constant(0.1),
                       )(chars)
    pool_char = CharLevelPooling()(char_conv)

    # input representation
    input_repre = Concatenate()([words, pos1, pos2, tags, pool_char])
    input_repre = Dropout(DROPOUT)(input_repre)

    # attention input
    mlp1 = Dense(ATT_HIDDEN_LAYER, activation="tanh")
    mlp2 = Dense(1, activation="softmax")
    alpha1 = attention(mlp1, mlp2, e1_pooled, words)
    alpha2 = attention(mlp1, mlp2, e2_pooled, words)
    alpha = Average()([alpha1, alpha2])
    alpha = RepeatVector(WORD_REPRE_SIZE)(alpha)
    alpha = Permute([2, 1])(alpha)
    input_repre = Multiply()([input_repre, alpha])

    # word-level convolution
    input_conved = Conv1D(filters=NB_FILTERS_WORD,
                          kernel_size=WINDOW_SIZE_WORD,
                          padding="same",
                          activation="relu",
                          kernel_initializer=TruncatedNormal(stddev=0.1),
                          bias_initializer=Constant(0.1))(input_repre)
    input_pooled = GlobalMaxPool1D()(input_conved)

    # fully connected
    output = Concatenate()([input_pooled, e1context, e2context])
    output = Dropout(DROPOUT)(output)
    output = Dense(
        units=NB_RELATIONS,
        activation="softmax",
        kernel_initializer=TruncatedNormal(stddev=0.1),
        bias_initializer=Constant(0.1),
        kernel_regularizer='l2',
        bias_regularizer='l2',
    )(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, tags_input, chars_input, e1_input, e2_input, e1context_input, e2context_input], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer='adam')
    # model.summary()
    return model


def attention(mlp1, mlp2, e_pooled, words):
    e_pooled = RepeatVector(SEQUENCE_LEN)(e_pooled)
    h = Concatenate()([words, e_pooled])
    u = mlp1(h)
    alpha = mlp2(u)
    alpha = Reshape([SEQUENCE_LEN])(alpha)
    return alpha


class CharLevelPooling(Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[3]

    def call(self, inputs, **kwargs):
        return K.max(inputs, axis=2)


class EntityContext(Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 3*WORD_EMBED_SIZE

    def call(self, inputs, **kwargs):
        mid, arounds = inputs
        first = K.reshape(arounds[:, 0, :], [-1, WORD_EMBED_SIZE])
        last = K.reshape(arounds[:, 1, :], [-1, WORD_EMBED_SIZE])
        context = K.concatenate([first, mid, last])
        return context


if __name__ == "__main__":
    word_embeddings = np.random.random([10, WORD_EMBED_SIZE])
    position_embeddings_1 = np.random.random([10, POSITION_EMBED_SIZE])
    position_embeddings_2 = np.random.random([10, POSITION_EMBED_SIZE])
    char_embeddings = np.random.random([10, CHAR_EMBED_SIZE])
    tag_embeddings = np.random.random([10, TAG_EMBED_SIZE])
    embeddings = make_dict(word_embeddings, position_embeddings_1, position_embeddings_2, char_embeddings, tag_embeddings)
    build_model(embeddings)
