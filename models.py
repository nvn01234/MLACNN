from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Reshape, Layer, RepeatVector, Permute, Multiply, Average, Subtract
from keras.engine import Model
from keras import backend as K
from keras import  initializers, regularizers

def build_model():
    # input
    words_input = Input(shape=[SEQUENCE_LEN, WORD_EMBED_SIZE], dtype='float32')
    pos1_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], dtype='float32')
    pos2_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], dtype='float32')
    e1_input = Input(shape=[WORD_EMBED_SIZE], dtype='float32')
    e2_input = Input(shape=[WORD_EMBED_SIZE], dtype='float32')

    input_repre = Concatenate()([words_input, pos1_input, pos2_input])
    input_repre = Dropout(DROPOUT)(input_repre)

    # attention
    alpha = input_attention(words_input, e1_input,e2_input)
    input_repre = Multiply()([input_repre, alpha])

    # convolution
    conv = []
    for size in WINDOW_SIZES:
        conv.append(Conv1D(filters=NB_FILTERS,
                           kernel_size=size,
                           padding="same",
                           activation="tanh")(input_repre))
    conv = Average()(conv)
    pool = GlobalMaxPool1D()(conv)

    e = entity_description(e1_input, e2_input)

    # fully connected
    output = Concatenate()([pool, e])
    output = Dropout(DROPOUT)(output)
    output = Dense(units=NB_RELATIONS, activation="softmax")(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer='adam')
    # model.summary()
    return model

def input_attention(words_input, e1_input, e2_input):
    e1 = RepeatVector(SEQUENCE_LEN)(e1_input)
    h1 = Concatenate()([words_input, e1])
    u1 = Dense(1, activation="tanh")(h1)
    alpha1 = Dense(1, activation="softmax")(u1)
    alpha1 = Reshape([SEQUENCE_LEN])(alpha1)
    alpha1 = RepeatVector(WORD_REPRE_SIZE)(alpha1)
    alpha1 = Permute([2, 1])(alpha1)

    e2 = RepeatVector(SEQUENCE_LEN)(e2_input)
    h2 = Concatenate()([words_input, e2])
    u2 = Dense(1, activation="tanh")(h2)
    alpha2 = Dense(1, activation="softmax")(u2)
    alpha2 = Reshape([SEQUENCE_LEN])(alpha2)
    alpha2 = RepeatVector(WORD_REPRE_SIZE)(alpha2)
    alpha2 = Permute([2, 1])(alpha2)

    alpha = Average()([alpha1, alpha2])
    return alpha

def entity_description(e1_input, e2_input):
    v = Subtract()([e1_input, e2_input])
    return v

if __name__ == "__main__":
    model = build_model()