from settings import *
from keras.layers import Input, Concatenate, Conv1D, GlobalMaxPool1D, Dense, Dropout, Reshape
from keras.engine import Model
from keras.optimizers import Adam


def build_model():
    words_input = Input(shape=[SEQUENCE_LEN, WORD_EMBED_SIZE], dtype='float32')
    pos1_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], dtype='float32')
    pos2_input = Input(shape=[SEQUENCE_LEN, POSITION_EMBED_SIZE], dtype='float32')
    e1_input = Input(shape=[WORD_EMBED_SIZE], dtype='float32')
    e2_input = Input(shape=[WORD_EMBED_SIZE], dtype='float32')

    input_repre = Concatenate()([words_input, pos1_input, pos2_input])
    input_repre = Dropout(rate=DROPOUT)(input_repre)

    pooled_output = []
    for window_size in WINDOW_SIZES:
        conv = Conv1D(filters=NB_FILTERS, kernel_size=window_size, padding="same", activation="tanh")(input_repre)
        pool = GlobalMaxPool1D()(conv)
        pooled_output.append(pool)
    output = Concatenate(name="concat")([*pooled_output, e1_input, e2_input])
    output = Dropout(rate=DROPOUT)(output)
    output = Dense(units=NB_RELATIONS, activation="softmax")(output)

    model = Model(inputs=[words_input, pos1_input, pos2_input, e1_input, e2_input], outputs=[output])
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    model.summary()
    return model
