import json

use_cuda = 0
learning_rate = 0
model_name = 0
checkpoint_path = 0
vocabulary_path = 0
batch_size = 0
hidden_size = 0
attn_method = 0
n_decoder_layers = 0
dropout = 0
max_length = 0
prefix = 0
n_encoder_layers = 0
n_decoder_layers = 0
clip = 0
print_every = 0
save_every = 0

def parse(config_str):
    global use_cuda
    global model_name
    global checkpoint_path
    global vocabulary_path
    global batch_size
    global hidden_size
    global attn_method
    global n_decoder_layers
    global n_encoder_layers
    global dropout
    global max_length
    global prefix
    global learning_rate
    global clip
    global print_every
    global save_every


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
        attn_method = config['MODEL']['ATTN_METHOD']
        n_encoder_layers = config['MODEL']['N_ENCODER_LAYERS']
        n_decoder_layers = config['MODEL']['N_DECODER_LAYERS']
        dropout = config['MODEL']['DROPOUT']

        n_iters = config['TRAIN']['N_EPOCHS']
        batch_size = config['TRAIN']['BATCH_SIZE']
        clip = config['TRAIN']['CLIP']
        learning_rate = config['TRAIN']['LEARNING_RATE']
        teacher_forcing_ratio = config['TRAIN']['TEACHER_FORCING_RATIO']

        print_every = config['TRAIN']['PRINT_EVERY']
        save_every = config['TRAIN']['SAVE_EVERY']

