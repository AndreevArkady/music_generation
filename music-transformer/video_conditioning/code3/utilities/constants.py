import torch

from preprocessing.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_VEL, RANGE_TIME_SHIFT

SEPERATOR = '========================='

# Taken from the paper
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.98
ADAM_EPSILON = 10e-9

LR_DEFAULT_START = 1.0
SCHEDULER_WARMUP_STEPS = 4000

TOKEN_END = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD = TOKEN_END + 1
VOCAB_SIZE = TOKEN_PAD + 1

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int32

TORCH_LABEL_TYPE = torch.long  # long is necessary for cross_entropy_loss
ADDITIONAL_FEATURES_TYPE = torch.long
MAX_TOKENS_SEQUENCE_LENGTH = 4000

PREPEND_ZEROS_WIDTH = 4

GENRES_LIST = ( # noqa WPS317
    'pop', 'jazz', 'rock', 'blues', 'classical', 'country',
    'soul', 'rap', 'latin', 'folk', 'electro', '[UNK]',
)
SENTIMENTS_LIST = ('minor', 'major')

CSV_HEADER = ('Epoch', 'Learn rate', 'Avg Train loss', 'Train Accuracy', 'Avg Eval loss', 'Eval accuracy')
