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
    e1_flat = Flatten()(e1)
    e2_flat = Flatten()(e2)
    e1context_flat = Flatten()(e1context)
    e2context_flat = Flatten()(e2context)

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
    pooled_char = []
    for size in WINDOW_SIZES_CHAR:
        chars = Conv2D(filters=NB_FILTERS_CHAR,
                       kernel_size=(1, size),
                       padding="same",
                       activation="relu",
                       kernel_initializer=TruncatedNormal(stddev=0.1),
                       bias_initializer=Constant(0.1),
                       )(chars)
        pool = CharLevelPooling()(chars)
        pooled_char.append(pool)

    # input representation
    input_repre = Concatenate()([words, pos1, pos2, tags, *pooled_char])
    input_repre = Dropout(DROPOUT)(input_repre)

    # attention input
    mlp_1 = Dense(units=ATT_HIDDEN_LAYER, activation="tanh")
    mlp_2 = Dense(units=1, activation="softmax")
    att_context_1 = attention_context(mlp_1, mlp_2, words, e1_flat)
    att_context_2 = attention_context(mlp_1, mlp_2, words, e2_flat)

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
    output = Concatenate()([*pooled_word, e1_flat, e2_flat, e1context_flat, e2context_flat, att_context_1, att_context_2])
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


def attention_context(mlp1, mlp2, words, e_flat):
    e_repeat = RepeatVector(SEQUENCE_LEN)(e_flat)
    h = Concatenate()([words, e_repeat])
    u = mlp1(h)
    alpha = mlp2(u)
    alpha = Reshape([SEQUENCE_LEN])(alpha)
    alpha = RepeatVector(WORD_EMBED_SIZE)(alpha)
    alpha = Permute([2, 1])(alpha)
    context = Multiply()([words, alpha])
    context = Sum()(context)
    return context


class Sum(Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        return K.sum(inputs, 1)


class CharLevelPooling(Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[3]

    def call(self, inputs, **kwargs):
        return K.max(inputs, axis=2)


if __name__ == "__main__":
    word_embeddings = np.random.random([10, WORD_EMBED_SIZE])
    position_embeddings_1 = np.random.random([10, POSITION_EMBED_SIZE])
    position_embeddings_2 = np.random.random([10, POSITION_EMBED_SIZE])
    char_embeddings = np.random.random([10, CHAR_EMBED_SIZE])
    tag_embeddings = np.random.random([10, TAG_EMBED_SIZE])
    embeddings = make_dict(word_embeddings, position_embeddings_1, position_embeddings_2, char_embeddings, tag_embeddings)
    build_model(embeddings)
