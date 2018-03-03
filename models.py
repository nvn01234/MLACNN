from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout
from keras.engine import Model
from keras.optimizers import Adam


def build_model():
    words_input = Input(shape=[SEQUENCE_LEN, WORD_EMBED_SIZE], name="word_input", dtype='float32')
    pos1_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], name="pos1_input", dtype='float32')
    pos2_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], name="pos2_input", dtype='float32')
    input_repre = Concatenate(name="input_representation")([words_input, pos1_input, pos2_input])
    conv = Conv1D(filters=NB_FILTERS, kernel_size=WINDOW_SIZE, padding="same", activation="tanh", name="convolution")(input_repre)
    pool = GlobalMaxPool1D(name="max_pooling")(conv)
    output = Dropout(name="dropout", rate=DROPOUT)(pool)
    output = Dense(name="output", units=NB_RELATIONS)(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input], outputs=[output])
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    model.summary()
    return model
