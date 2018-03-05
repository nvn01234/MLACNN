from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Reshape, Layer, RepeatVector, Permute, Multiply, Average, Subtract, Embedding, Flatten
from keras.engine import Model
from keras import backend as K
from keras import  initializers, regularizers
from keras.initializers import TruncatedNormal, Constant
from keras.optimizers import Adam

def build_model_2():
    # input
    words_input = Input(shape=[SEQUENCE_LEN, WORD_EMBED_SIZE], dtype='float32')
    pos1_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    pos2_input = Input(shape=[SEQUENCE_LEN], dtype='int32')
    e1_input = Input(shape=[3, WORD_EMBED_SIZE], dtype='float32')
    e2_input = Input(shape=[3, WORD_EMBED_SIZE], dtype='float32')

    e1_flat = Flatten()(e1_input)
    e2_flat = Flatten()(e2_input)
    pos1_embed = Embedding(NB_DISTANCES, POSITION_EMBED_SIZE)(pos1_input)
    pos2_embed = Embedding(NB_DISTANCES, POSITION_EMBED_SIZE)(pos2_input)

    input_repre = Concatenate()([words_input, pos1_embed, pos2_embed])
    input_repre = Dropout(DROPOUT)(input_repre)

    # attention
    alpha = input_attention(words_input, e1_flat, e2_flat)
    input_repre = Multiply()([input_repre, alpha])

    pooled = conv_maxpool(input_repre)
    e = entities_features(e1_flat, e2_flat)
    output = MLP([pooled, e])

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    optimizer = Adam(LEARNING_RATE)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    # model.summary()
    return model

def build_model():
    # input
    words_input = Input(shape=[SEQUENCE_LEN, WORD_EMBED_SIZE], dtype='float32')
    pos1_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], dtype='float32')
    pos2_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], dtype='float32')
    e1_input = Input(shape=[3, WORD_EMBED_SIZE], dtype='float32')
    e2_input = Input(shape=[3, WORD_EMBED_SIZE], dtype='float32')

    e1_flat = Flatten()(e1_input)
    e2_flat = Flatten()(e2_input)

    input_repre = Concatenate()([words_input, pos1_input, pos2_input])
    input_repre = Dropout(DROPOUT)(input_repre)

    # attention
    alpha = input_attention(words_input, e1_flat, e2_flat)
    input_repre = Multiply()([input_repre, alpha])

    pooled = conv_maxpool(input_repre)
    e = entities_features(e1_flat, e2_flat)
    output = MLP([pooled, e])

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    optimizer = Adam(LEARNING_RATE)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    # model.summary()
    return model

def input_attention(words_input, e1_input, e2_input):
    e1 = RepeatVector(SEQUENCE_LEN)(e1_input)
    h1 = Concatenate()([words_input, e1])
    u1 = Dense(WORD_EMBED_SIZE, activation="tanh")(h1)
    alpha1 = Dense(1, activation="softmax")(u1)
    alpha1 = Reshape([SEQUENCE_LEN])(alpha1)
    alpha1 = RepeatVector(WORD_REPRE_SIZE)(alpha1)
    alpha1 = Permute([2, 1])(alpha1)

    e2 = RepeatVector(SEQUENCE_LEN)(e2_input)
    h2 = Concatenate()([words_input, e2])
    u2 = Dense(WORD_EMBED_SIZE, activation="tanh")(h2)
    alpha2 = Dense(1, activation="softmax")(u2)
    alpha2 = Reshape([SEQUENCE_LEN])(alpha2)
    alpha2 = RepeatVector(WORD_REPRE_SIZE)(alpha2)
    alpha2 = Permute([2, 1])(alpha2)

    alpha = Average()([alpha1, alpha2])
    return alpha

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
    output = Dense(
        units=NB_RELATIONS,
        activation="softmax"
    )(output)
    return output

if __name__ == "__main__":
    build_model()