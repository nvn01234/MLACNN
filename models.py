from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout
from keras.engine import Model
from keras.optimizers import Adam


def build_model():
    words_input = Input(shape=[SEQUENCE_LEN, WORD_EMBED_SIZE], name="word_input", dtype='float32')
    pos1_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], name="pos1_input", dtype='float32')
    pos2_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], name="pos2_input", dtype='float32')
    e1_input = Input(shape=[WORD_EMBED_SIZE], name="e1_input", dtype='float32')
    e2_input = Input(shape=[WORD_EMBED_SIZE], name="e2_input", dtype='float32')

    input_repre = Concatenate(name="input_representation")([words_input, pos1_input, pos2_input])

    pooled_output = []
    for window_size in [3, 4, 5]:
        conv = Conv1D(filters=NB_FILTERS, kernel_size=window_size, padding="same", activation="tanh")(input_repre)
        pool = GlobalMaxPool1D(name="max_pooling")(conv)
        pooled_output.append(pool)
    conv = Concatenate(name="convolution_concat")(pooled_output)
    output = Concatenate(name="concat")([conv, e1_input, e2_input])
    output = Dropout(name="dropout", rate=DROPOUT)(output)
    output = Dense(name="output", units=NB_RELATIONS)(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer='sgd')
    model.summary()
    return model
