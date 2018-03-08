# character-level convolution
NB_FILTERS_CHAR = 50
WINDOW_SIZE_CHAR = 3
CHAR_EMBED_SIZE = 300
WORD_LEN = 24

# word-level convolution
NB_FILTERS_WORD = 200  # hidden layer 1
WINDOW_SIZE_WORD = 3
WORD_EMBED_SIZE = 300
SEQUENCE_LEN = 98

# position feature
POSITION_EMBED_SIZE = 5
MAX_DISTANCE = 61
MIN_DISTANCE = -61
NB_POSITIONS = MAX_DISTANCE - MIN_DISTANCE + 1

ENTITY_LEN = 4

TAG_EMBED_SIZE = 10

NB_RELATIONS = 9 * 2 + 1

DROPOUT = 0.5

NB_EPOCHS = 200
BATCH_SIZE = 100

ATT_HIDDEN_LAYER = 50
WORD_REPRE_SIZE = WORD_EMBED_SIZE + 2*POSITION_EMBED_SIZE + TAG_EMBED_SIZE + NB_FILTERS_CHAR