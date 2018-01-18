import json

use_cuda = 0
learning_rate = 0
model_name = 0
checkpoint_path = 0
vocabulary_path = 0
batch_size = 0
hidden_size = 0
dropout = 0
max_length = 0
prefix = 0
n_encoder_layers = 0
n_decoder_layers = 0
clip = 0
print_every = 0
save_every = 0
encoder_bidirectional = 0
single_embedding = 0
reverse_input = 0
data_path = 0
dialogue_corpus = 0
min_length = 0
max_length = 0
min_count = 0
ckpt_epoch = 0
use_attn = 0
movie_conversations = 0
movie_lines = 0
is_augment = 0
vocabulary_size = 0

def parse(config_str):
    global use_cuda
    global model_name
    global checkpoint_path
    global vocabulary_path
    global batch_size
    global hidden_size
    global n_decoder_layers
    global n_encoder_layers
    global encoder_bidirectional
    global dropout
    global max_length
    global prefix
    global learning_rate
    global clip
    global print_every
    global save_every
    global single_embedding
    global reverse_input
    global data_path
    global dialogue_corpus
    global movie_conversations
    global movie_lines
    global is_augment
    global min_length
    global max_length
    global min_count
    global ckpt_epoch
    global use_attn
    global vocabulary_size

    with open('config/%s.json' %(config_str)) as config_file:
        config = json.load(config_file)
        use_cuda = config['TRAIN']['CUDA']
        model_name = config['MODEL']['NAME']
        checkpoint_path = config['TRAIN']['PATH']
        vocabulary_path = '%s%s' % (checkpoint_path,\
                config['TRAIN']['VOCABULARY'])
        batch_size = config['TRAIN']['BATCH_SIZE']
        max_length=config['LOADER']['MAX_LENGTH']
        prefix = config['TRAIN']['PREFIX']

        hidden_size = config['MODEL']['HIDDEN_SIZE']
        n_encoder_layers = config['MODEL']['N_ENCODER_LAYERS']
        n_decoder_layers = config['MODEL']['N_DECODER_LAYERS']
        dropout = config['MODEL']['DROPOUT']

        n_iters = config['TRAIN']['N_EPOCHS']
        batch_size = config['TRAIN']['BATCH_SIZE']
        clip = config['TRAIN']['CLIP']
        learning_rate = config['TRAIN']['LEARNING_RATE']
        teacher_forcing_ratio = config['TRAIN']['TEACHER_FORCING_RATIO']
        vocabulary_size = config['TRAIN']['VOCABULARY_SIZE']

        print_every = config['TRAIN']['PRINT_EVERY']
        save_every = config['TRAIN']['SAVE_EVERY']

        encoder_bidirectional = config['MODEL']['ENCODER_BIDIRECTIONAL']
        single_embedding = config['MODEL']['SINGLE_EMBEDDING']
        use_attn = config['MODEL']['USE_ATTN']
        reverse_input = config['TRAIN']['REVERSE_INPUT']

        data_path = config['DATA']['PATH']
        dialogue_corpus = config['DATA']['DIALOGUE_CORPUS']
        movie_conversations = config['DATA']['MOVIE_CONVERSATIONS']
        movie_lines = config['DATA']['MOVIE_LINES']
        is_augment = config['DATA']['IS_AUGMENT']
        # range of sentenct length
        min_length = config['LOADER']['MIN_LENGTH']
        max_length = config['LOADER']['MAX_LENGTH']
        # least word count
        min_count = config['LOADER']['MIN_COUNT']
        ckpt_epoch = config['TEST']['CKPT_EPOCH']
