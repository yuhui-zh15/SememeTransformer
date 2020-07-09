import os
import sys
import json
import argparse
import numpy as np
import random
import torch
from data import get_data, pad_batch, batchify, Batch
from transformer import NoamOpt, make_transformer_model, make_lstm_model
import logging
from sklearn import metrics

parser = argparse.ArgumentParser(description='Sememe')
# paths
parser.add_argument("--corpus", type=str, default='sage', help="sage|csu|pp")
parser.add_argument("--hypes", type=str, default=None, help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='exp/', help="Output directory")
parser.add_argument("--inputdir", type=str, default='', help="Input model dir")
parser.add_argument("--cut_down_len", type=int, default=128, help="sentence will be cut down if tokens num greater than this")
# training
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--bptt_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dpout", type=float, default=0.2, help="residual, embedding, attention dropout") # 3 dropouts
parser.add_argument("--warmup_steps", type=int, default=8000, help="OpenNMT uses steps") # TransformerLM uses 0.2% of training data as warmup step, that's 5785 for DisSent5/8, and 8471 for DisSent-All
parser.add_argument("--factor", type=float, default=1.0, help="learning rate scaling factor")
parser.add_argument("--l2", type=float, default=0.01, help="on non-bias non-gain weights")
parser.add_argument("--max_norm", type=float, default=2., help="max norm (grad clipping). Original paper uses 1.")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")
parser.add_argument('--lm_coef', type=float, default=0.5)
parser.add_argument("--train_emb", default=False, action='store_true', help="Allow to learn embedding, default to False")
parser.add_argument("--init_emb", default=False, action='store_true', help="Initialize embedding randomly, default to False")
parser.add_argument("--pick_hid", default=True, action='store_true', help="Pick correct hidden states")
parser.add_argument("--tied", default=False, action='store_true', help="Tie weights to embedding, should be always flagged True")
parser.add_argument("--model_type", type=str, default="transformer", help="transformer|lstm|caml")
parser.add_argument("--ratio", type=float, default=1.0, help="percent of data used for training")
# model
parser.add_argument("--d_ff", type=int, default=2048, help="decoder nhid dimension")
parser.add_argument("--d_model", type=int, default=768, help="decoder nhid dimension")
parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads")
parser.add_argument("--n_layers", type=int, default=6, help="decoder num layers")
parser.add_argument("--n_lstm_layers", type=int, default=1, help="decoder num lstm layers")
parser.add_argument("--n_kernels", type=int, default=50, help="caml kernel number")
parser.add_argument("--kernel_size", type=int, default=4, help="caml kernel size")
# gpu
parser.add_argument("--seed", type=int, default=1234, help="seed")

# ------------------------------------------------------------------------------
# python3 trainer.py --outputdir exp/test --corpus csu --hypes hypes/weibo_new.json 
# --model_type transformer --tied --inputdir exp/weibo_new_nosememe_mix/model-10.pickle
# ------------------------------------------------------------------------------
parser.add_argument('--dataset', type=str, choices=['headline', 'weibo', 'lcqmc'])
parser.add_argument("--sememe", default=False, action='store_true', help="sememe embedding")
parser.add_argument('--epoch', type=int, help='The epoch of model loaded')
parser.add_argument('--percent', type=int, help='The percent of training data', default=100)
params, _ = parser.parse_known_args()

params.outputdir = 'exp/test'
params.corpus = 'csu'
parser.model_type = 'transformer'
parser.tied = True
if params.hypes is None:
    params.hypes = {
        'headline': 'hypes/headline.json',
        'weibo': 'hypes/weibo_new.json',
        'lcqmc': 'hypes/lcqmc.json'
    }[params.dataset]
percent = ('%d_' % params.percent) if params.percent < 100 else ''
params.inputdir = {
    ('headline', False): 'exp/headline_%snosememe' % percent,
    ('headline', True): 'exp/headline_%ssememe_avg_multitask_loss_lm' % percent,
    ('weibo', False): 'exp/weibo_%snew_nosememe_mix' % percent,
    ('weibo', True): 'exp/weibo_%snew_nosememe_mix_sememe_multitask' % percent,
    ('lcqmc', False): 'exp/lcqmc_%snosememe' % percent,
    ('lcqmc', True): 'exp/lcqmc_%ssememe_avg_multitask_loss_lm' % percent
}[params.dataset, params.sememe] + ('/model-%d.pickle' % params.epoch)
params.label2id = {
    'headline': {
        "essay": 17, 
        "story": 15, 
        "fashion": 13, 
        "finance": 12, 
        "entertainment": 11, 
        "food": 10, 
        "car": 9, 
        "travel": 8, 
        "regimen": 16, 
        "sports": 7, 
        "society": 6, 
        "game": 5, 
        "tech": 4, 
        "world": 3, 
        "baby": 2, 
        "military": 1, 
        "discovery": 14, 
        "history": 0
    }, 
    'weibo': {
        'like': 0, 
        'disgust': 1, 
        'happiness': 2, 
        'sadness': 3, 
        'anger': 4, 
        'surprise': 5, 
        'fear': 6
    },
    'lcqmc': {
        'same': 1,
        'diff': 0
    }
}[params.dataset]

"""
SEED
"""
random.seed(params.seed)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
Logging
"""
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)
if not os.path.exists(params.outputdir): os.makedirs(params.outputdir)
file_handler = logging.FileHandler("{0}/log.txt".format(params.outputdir))
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)
logger.info('\nTogrep : {0}\n'.format(sys.argv[1:]))
logger.info(params)

"""
Default json file loading
"""
json_config = json.load(open(params.hypes))
data_dir = json_config['data_dir']
prefix = json_config[params.corpus]
encoder_path = json_config['encoder_path']
label_size = json_config['label_size']
if params.init_emb: wordvec_path = json_config['wordvec_path']

"""
BPE encoder
"""
encoder = json.load(open(encoder_path))
encoder['_pad_'] = len(encoder)
encoder['_start_'] = len(encoder)
encoder['_end_'] = len(encoder)
encoder['_unk_'] = len(encoder)
n_special = 4

"""
DATA
"""
train, valid, test = get_data(encoder, data_dir, prefix, params.cut_down_len, label_size, params.ratio) 
max_len = 0.
if params.corpus == 'sage':
    train['text'] = batchify(np.array(train['text'][0]), params.batch_size)
    valid['text'] = batchify(np.array(valid['text'][0]), params.batch_size)
    test['text'] = batchify(np.array(test['text'][0]), params.batch_size)

"""
Params
"""
if params.init_emb:
    word_embeddings = np.concatenate([np.load(wordvec_path).astype(np.float32),
                                      np.zeros((1, params.d_model), np.float32), # pad, zero-value!
                                      (np.random.randn(n_special - 1, params.d_model) * 0.02).astype(np.float32)], 0)
else:                                                          
    word_embeddings = None


"""
MODEL
"""
# model config
config_model = {
    'n_words': len(encoder),
    'd_model': params.d_model, # same as word embedding size
    'd_ff': params.d_ff, # this is the bottleneck blowup dimension
    'n_layers': params.n_layers,
    'dpout': params.dpout,
    'bsize': params.batch_size,
    'n_classes': label_size,
    'n_heads': params.n_heads,
    'train_emb': params.train_emb,
    'init_emb': params.init_emb,
    'pick_hid': params.pick_hid,
    'tied': params.tied,
    'n_lstm_layers': params.n_lstm_layers,
    'n_kernels': params.n_kernels,
    'kernel_size': params.kernel_size,
    'sememe_path': json_config['sememe_path'],
    'sememe_size': np.load(json_config['sememe_path']).shape[1]
}

if params.model_type == "lstm":
    logger.info('model lstm')
    model = make_lstm_model(encoder, config_model, word_embeddings)
else:
    logger.info('model transformer')
    model = make_transformer_model(encoder, config_model, word_embeddings, params.sememe)
logger.info(model)
need_grad = lambda x: x.requires_grad
model_opt = NoamOpt(params.d_model, params.factor, params.warmup_steps, torch.optim.Adam(filter(need_grad, model.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
model.cuda()

def evaluate_epoch_csu(epoch, eval_type='valid'):
    # initialize
    logger.info('\n{} : Epoch {}'.format(eval_type.upper(), epoch))
    model.eval()
    
    # data without shuffle
    if eval_type == 'train': text, label = train['text'], train['label']
    elif eval_type == 'valid': text, label = valid['text'], valid['label']
    else: 
        if params.dataset == 'headline':
            text, label = valid['text'], valid['label']
        else:
            text, label = test['text'], test['label']

    valid_preds, valid_labels = [], []

    for stidx in range(0, len(text), params.batch_size):
        # prepare batch
        text_batch = pad_batch(text[stidx: stidx + params.batch_size].tolist(), encoder, pad_start_end=True)
        label_batch = label[stidx: stidx + params.batch_size]
        
        b = Batch(text_batch, label_batch, encoder['_pad_'])

        # model forward
        clf_output = model(b, clf=True, lm=False)

        # evaluation
        pred = clf_output.max(1)[1].data.cpu().numpy().astype(float)
        valid_preds.extend(pred.tolist())
        valid_labels.extend(label_batch.tolist())
        
    valid_preds, valid_labels = np.array(valid_preds), np.array(valid_labels)
    A = (valid_preds == valid_labels).astype(float)
    acc = A.mean()

    runid = params.inputdir.replace('exp/', '').replace('/', '-')
    save_path = 'exp/adv/acc.%s.npy' % runid
    print('Saved ACC to:', save_path)
    np.save(save_path, A)

    logger.info('{}; acc {}'.format(
        epoch, 
        round(acc, 3),
    ))


word2sememe = np.load(config_model['sememe_path'])


if __name__ == '__main__':
    epoch = 1
    if params.corpus == 'csu':
        del model
        model = torch.load(params.inputdir)
        print(params.inputdir, params.hypes)
        evaluate_epoch_csu(epoch, eval_type='test')
    else:
        raise ValueError('wrong corpus')
