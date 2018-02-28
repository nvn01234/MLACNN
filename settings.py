WORD_EMBED_SIZE = 300
POSITION_EMBED_SIZE = 25
WORD_REPRE_SIZE = WORD_EMBED_SIZE + 2 * POSITION_EMBED_SIZE

SEQUENCE_LEN = 100
MAX_DISTANCE = 61
MIN_DISTANCE = -61
NB_DISTANCES = MAX_DISTANCE - MIN_DISTANCE + 1
NB_RELATIONS = 9 * 2 + 1

NB_FILTERS = 1000
WINDOW_SIZES = [3, 4, 5]

LEARNING_RATE = 1e-3
DROPOUT = 0.5
L2_REG_LAMBDA = 0

NB_EPOCHS = 100
BATCH_SIZE = 50

TEST_ONLY = False
SAVE_PATH = None  # save model
LOG_FILE = None

# origin data
TRAIN_FILE = "origin_data/TRAIN_FILE.TXT"
TEST_FILE = "origin_data/TEST_FILE_FULL.TXT"
ORIGIN_WORD_EMBEDDINGS_PATH = "origin_data/glove.6B.300d.txt"

# initialized data
WORD_EMBEDDINGS_PATH = "data/word_embeddings.npy"
X_WORDS_TRAIN_PATH = "data/x_words_train.npy"
X_POS1_TRAIN_PATH = "data/x_pos1_train.npy"
X_POS2_TRAIN_PATH = "data/x_pos2_train.npy"
X_E1_TRAIN_PATH = "data/x_e1_train.npy"
X_E2_TRAIN_PATH = "data/x_e2_train.npy"
Y_TRAIN_PATH = "data/y_train.npy"
X_WORDS_TEST_PATH = "data/x_words_test.npy"
X_POS1_TEST_PATH = "data/x_pos1_test.npy"
X_POS2_TEST_PATH = "data/x_pos2_test.npy"
X_E1_TEST_PATH = "data/x_e1_test.npy"
X_E2_TEST_PATH = "data/x_e2_test.npy"
Y_TEST_PATH = "data/y_test.npy"