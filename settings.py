NB_FILTERS = 100
WINDOW_SIZES = [3,4,5]

WORD_EMBED_SIZE = 300
POSITION_EMBED_SIZE = 25
WORD_REPRE_SIZE = WORD_EMBED_SIZE + 2 * POSITION_EMBED_SIZE

SEQUENCE_LEN = 97
MAX_DISTANCE = 61
MIN_DISTANCE = -61
NB_DISTANCES = MAX_DISTANCE - MIN_DISTANCE + 1
NB_RELATIONS = 9 * 2 + 1

HIDDEN_LAYER = 100
DROPOUT = 0.5

NB_EPOCHS = 10
BATCH_SIZE = 50