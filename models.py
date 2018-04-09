from settings import *
from keras.layers import Input, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, RepeatVector, Reshape, \
    Multiply, Permute, GlobalMaxPool1D
from keras.engine import Model, Layer
from keras import backend as K
import numpy as np
from keras.initializers import TruncatedNormal, Constant
from metrics import evaluate

from utils import make_dict


def build_model(embeddings, lexical_feature=None, attention_input=False, piecewise_max_pool=False):
    # input representation features
    words_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    pos1_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    pos2_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    segs_input = Input(shape=[SEQUENCE_LEN, 3], dtype='float32')

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

    # input representation
    input_repre = Concatenate()([words, pos1, pos2])
    input_repre = Dropout(DROPOUT)(input_repre)

    # input attention
    if attention_input:
        e1_conved = Conv1D(filters=WORD_EMBED_SIZE,
                           kernel_size=ENTITY_LEN,
                           padding="valid",
                           activation="relu",
                           kernel_initializer=TruncatedNormal(stddev=0.1),
                           bias_initializer=Constant(0.1))(e1)
        e1_conved = Reshape([WORD_EMBED_SIZE])(e1_conved)
        e1_repeat = RepeatVector(SEQUENCE_LEN)(e1_conved)
        e2_conved = Conv1D(filters=WORD_EMBED_SIZE,
                           kernel_size=ENTITY_LEN,
                           padding="valid",
                           activation="relu",
                           kernel_initializer=TruncatedNormal(stddev=0.1),
                           bias_initializer=Constant(0.1))(e2)
        e2_conved = Reshape([WORD_EMBED_SIZE])(e2_conved)
        e2_repeat = RepeatVector(SEQUENCE_LEN)(e2_conved)
        concat = Concatenate()([words, e1_repeat, e2_repeat])
        alpha = Dense(1, activation="softmax")(concat)
        alpha = Reshape([SEQUENCE_LEN])(alpha)
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
    if piecewise_max_pool:
        input_pooled = PiecewiseMaxPool()([input_conved, segs_input])
    else:
        input_pooled = GlobalMaxPool1D()(input_conved)

    # fully connected
    if lexical_feature is None:
        output = input_pooled
    else:
        outputs = [input_pooled]
        if 1 in lexical_feature:
            outputs.append(e1_flat)
        if 2 in lexical_feature:
            outputs.append(e2_flat)
        if 3 in lexical_feature:
            outputs.append(e1context_flat)
        if 4 in lexical_feature:
            outputs.append(e2context_flat)
        output = Concatenate()(outputs)
    output = Dropout(DROPOUT)(output)
    output = Dense(
        units=NB_RELATIONS,
        activation="softmax",
        kernel_initializer=TruncatedNormal(stddev=0.1),
        bias_initializer=Constant(0.1),
        kernel_regularizer='l2',
        bias_regularizer='l2',
    )(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input, e1context_input, e2context_input, segs_input], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model


class CharLevelPooling(Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[3]

    def call(self, inputs, **kwargs):
        return K.max(inputs, axis=2)


class PiecewiseMaxPool(Layer):
    def compute_output_shape(self, input_shape):
        return None, PCNN_OUTPUT_SIZE

    def call(self, inputs, **kwargs):
        inputs, segments = inputs

        seg1 = inputs * K.expand_dims(segments[:, :, 0])
        seg2 = inputs * K.expand_dims(segments[:, :, 1])
        seg3 = inputs * K.expand_dims(segments[:, :, 2])

        output1 = K.expand_dims(K.max(seg1, 1))
        output2 = K.expand_dims(K.max(seg2, 1))
        output3 = K.expand_dims(K.max(seg3, 1))
        output = K.concatenate([output1, output2, output3])
        output = K.reshape(output, [-1, PCNN_OUTPUT_SIZE])
        return output


if __name__ == "__main__":
    word_embeddings = np.random.random([10, WORD_EMBED_SIZE])
    position_embeddings_1 = np.random.random([10, POSITION_EMBED_SIZE])
    position_embeddings_2 = np.random.random([10, POSITION_EMBED_SIZE])
    embeds = make_dict(word_embeddings, position_embeddings_1, position_embeddings_2)
    build_model(embeds)
