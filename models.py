from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Reshape, Layer, RepeatVector, Permute, Multiply, Average
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
    # input_repre = Dropout(DROPOUT)(input_repre)

    # attention
    e1 = RepeatVector(SEQUENCE_LEN)(e1_input)
    e2 = RepeatVector(SEQUENCE_LEN)(e2_input)
    h = Concatenate()([words_input, e1, e2])
    u = Dense(1, activation="tanh")(h)
    alpha = Dense(1, activation="softmax")(u)
    alpha = Reshape([SEQUENCE_LEN])(alpha)
    alpha = RepeatVector(WORD_REPRE_SIZE)(alpha)
    alpha = Permute([2,1])(alpha)
    input_repre = Multiply()([input_repre, alpha])

    # convolution
    conv = Conv1D(filters=NB_FILTERS,
                  kernel_size=WINDOW_SIZE,
                  padding="same",
                  activation="tanh")(input_repre)
    pool = GlobalMaxPool1D()(conv)

    # fully connected
    output = Dropout(DROPOUT)(pool)
    output = Dense(units=NB_RELATIONS, activation="softmax")(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer='adam')
    model.summary()
    return model



if __name__ == "__main__":
    model = build_model()