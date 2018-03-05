from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Reshape, Layer, RepeatVector, Permute, Multiply, Average, Subtract, Embedding, Flatten, SeparableConv1D, Conv2D, GlobalMaxPool2D
from keras.engine import Model, InputSpec
from keras import backend as K
from keras import  initializers, regularizers
from keras.initializers import TruncatedNormal, Constant
from keras.optimizers import Adam
import numpy as np

def build_model():
    # input
    words_input = Input(shape=[SEQUENCE_LEN, WORD_EMBED_SIZE], dtype='float32')
    pos1_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    pos2_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    e1_input = Input(shape=[3, WORD_EMBED_SIZE], dtype='float32')
    e2_input = Input(shape=[3, WORD_EMBED_SIZE], dtype='float32')
    chars_input = Input(shape=[SEQUENCE_LEN, WORD_LEN], dtype='int32')

    e1_flat = Flatten()(e1_input)
    e2_flat = Flatten()(e2_input)
    pos1_embed = Embedding(NB_DISTANCES, POSITION_EMBED_SIZE)(pos1_input)
    pos2_embed = Embedding(NB_DISTANCES, POSITION_EMBED_SIZE)(pos2_input)
    char_embeddings = np.load("data/char_embeddings.npy")
    char_embed = Embedding(char_embeddings.shape[0], char_embeddings.shape[1], weights=[char_embeddings], trainable=False)(chars_input)

    chars = char_level_word_feature(char_embed)
    input_repre = Concatenate()([words_input, pos1_embed, pos2_embed, chars])
    input_repre = Dropout(DROPOUT)(input_repre)

    pooled = conv_maxpool(input_repre)
    e = entities_features(e1_flat, e2_flat)
    output = MLP([pooled, e])

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input, chars_input], outputs=[output])
    optimizer = Adam(LEARNING_RATE)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    # model.summary()
    return model

def char_level_word_feature(chars):
    conv = Conv2D(filters=NB_FILTERS_CHAR,
                  kernel_size=(1, WINDOW_SIZE_CHAR),
                  padding="same",
                  activation="relu")(chars)
    pool = GlobalMaxPooling1D_2()(conv)
    return pool

def conv_maxpool(input_repre):
    pooled = []
    for size in WINDOW_SIZES:
        conv = Conv1D(filters=NB_FILTERS,
                      kernel_size=size,
                      padding="same",
                      activation="relu",
                      )(input_repre)
        wo = GlobalMaxPool1D()(conv)
        pooled.append(wo)
    pooled = Concatenate()(pooled)
    return pooled


def entities_features(e1_input, e2_input):
    e = Concatenate()([e1_input, e2_input])
    return e

def MLP(features):
    output = Concatenate()(features)
    output = Dropout(DROPOUT)(output)
    output = Dense(units=NB_RELATIONS, activation="softmax")(output)
    return output

class GlobalMaxPooling1D_2(GlobalMaxPool1D):
    def __init__(self, **kwargs):
        super(GlobalMaxPooling1D_2, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], SEQUENCE_LEN, NB_FILTERS_CHAR)

    def call(self, inputs):
        return K.max(inputs, axis=2)

if __name__ == "__main__":
    build_model()