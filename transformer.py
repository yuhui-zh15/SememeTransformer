import copy
import torch.nn as nn
import torch
import math
import logging
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab_size, np_word_embedding=None, word_embedding_weight=None):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        # this has the advantage that we don't need an embedding matrix actually...
        # only need this one...
        if np_word_embedding is not None:
            self.proj.weight.data.copy_(torch.from_numpy(np_word_embedding))
            self.proj.weight.requires_grad = False
        if word_embedding_weight is not None:
            self.proj.weight = word_embedding_weight  # tied-weights

    def forward(self, x):
        # return F.log_softmax(self.proj(x), dim=-1)
        return self.proj(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.-
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # can consider changing this non-linearity!
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# we will learn and predict via our own embeddings
class Embeddings(nn.Module):
    def __init__(self, encoder, config, word_embeddings=None):
        # encoder is the dictionary, not text_encoder
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(len(encoder), config['d_model'])
        self.d_model = config['d_model']
        if config['init_emb']:
            assert word_embeddings is not None
            logging.info('copy embeddings...')
            logging.info('2-norm %f' % (np.linalg.norm(word_embeddings)))
            self.lut.weight.data.copy_(torch.from_numpy(word_embeddings))
        if not config['train_emb']:
            self.lut.weight.requires_grad = False

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class SememeEmbeddings(nn.Module):
    def __init__(self, config):
        # encoder is the dictionary, not text_encoder
        super(SememeEmbeddings, self).__init__()
        self.word2sememe = np.load(config['sememe_path'])
        self.lut = nn.Embedding(config['sememe_size'], config['d_model'])
        self.d_model = config['d_model']

    def forward(self, x):
        sememe_onehot = torch.tensor(self.word2sememe[x.cpu().reshape(-1)].reshape([x.shape[0], x.shape[1], -1])).float().cuda()
        scalar = sememe_onehot.sum(2) + 1e-6
        sememe_embed = sememe_onehot.matmul(self.lut.weight)
        return sememe_embed * math.sqrt(self.d_model) / scalar.unsqueeze(2)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, config, max_len=5000):
        # ctx_embeddings: (max_len, n_embed)
        # we don't need to define new, just use the same...

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config['dpout'])

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, config['d_model'])
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., config['d_model'], 2) *
                             -(math.log(10000.0) / config['d_model']))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = torch.from_numpy(ctx_embeddings)
        pe = pe.unsqueeze(0)  # add one dimension to beginning (1, time, n_embed)
        self.register_buffer('pe', pe)  # this will add pe to self

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, config, decoder, tgt_embed, generator, sememe_embed=None, generator_sememe=None):
        super(Transformer, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.sememe_embed = sememe_embed
        self.generator = generator
        self.generator_sememe = generator_sememe
        self.config = config

        self.classifier = nn.Linear(config['d_model'], config['n_classes'])
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        # self.bce_loss = nn.BCEWithLogitsLoss(reduce=False)
        self.bce_loss = nn.MultiLabelSoftMarginLoss(reduce=False)
        self.dropout = nn.Dropout(config['dpout'])

    def encode(self, tgt, tgt_mask):
        # tgt, tgt_mask need to be on CUDA before being put in here
        embed = self.tgt_embed(tgt)
        if self.sememe_embed: embed += self.sememe_embed(tgt)
        embed = self.dropout(embed)
        return self.dropout(self.decoder(embed, tgt_mask))

    def pick_h(self, h, lengths):
        # batch_size, lengths
        corr_h = []
        for i, j in enumerate(lengths):
            corr_h.append(h[i, j-1, :])
        corr_h = torch.stack(corr_h, dim=0)
        return corr_h

    def forward(self, batch, clf=True, lm=True):
        "Take in and process masked src and target sequences."
        ret = []
        # this computes LM targets!! before the Generator
        u_h = self.encode(batch.text, batch.text_mask)
        if clf:
            # u_h, v_h: (batch_size, time_step, d_model) (which is n_embed)
            if self.config['pick_hid']: u = self.pick_h(u_h, batch.text_lengths)
            else: u = u_h[:, -1, :] # last hidden state
            clf_output = self.classifier(u)
            ret.append(clf_output)
        # compute LM
        if lm:
            text_y = self.generator(u_h)
            if self.sememe_embed: 
                sememe_y = self.generator_sememe(u_h)
                ret.append((text_y, sememe_y))
            else:
                ret.append(text_y)
        return ret[0] if len(ret) == 1 else ret

    def compute_clf_loss(self, logits, labels, multilabel=False):
        if multilabel:
            loss = self.bce_loss(logits, labels)
        else:
            loss = self.ce_loss(logits, labels)
        return loss.mean()

    def compute_lm_loss(self, text_h, text_y, text_loss_mask):
        loss = self.ce_loss(text_h.contiguous().view(-1, self.config['n_words']), text_y.view(-1)).view(text_h.size(0), -1)
        loss *= text_loss_mask  # mask sequence loss
        return loss.mean()


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.decoder = nn.LSTM(config['d_model'], config['d_model'], config['n_lstm_layers'], dropout=config['dpout'], batch_first=True)
        self.tgt_embed = nn.Embedding(config['n_words'], config['d_model'])
        self.generator = nn.Linear(config['d_model'], config['n_words'])
        if config['tied']:
            self.generator.weight = self.tgt_embed.weight
        self.classifier = nn.Linear(config['d_model'], config['n_classes'])
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        self.dropout = nn.Dropout(config['dpout'])
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.tgt_embed.weight.data.uniform_(-initrange, initrange)
        self.generator.bias.data.zero_()
        self.generator.weight.data.uniform_(-initrange, initrange)
        
    def encode(self, tgt, lengths):
        return self.autolen_rnn(self.tgt_embed(tgt), lengths)
    
    def autolen_rnn(self, inputs, lengths):
        inputs = self.dropout(inputs)
        idx = np.argsort(-lengths)
        revidx = np.argsort(idx)
        packed_emb = nn.utils.rnn.pack_padded_sequence(inputs[idx, :, :], lengths[idx], batch_first=True)
        output, (h, c) = self.decoder(packed_emb)
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        output = output[revidx, :, :]
        output = self.dropout(output)
        return output

    def pick_h(self, h, lengths):
        corr_h = []
        for i, j in enumerate(lengths): corr_h.append(h[i, j-1, :])
        corr_h = torch.stack(corr_h, dim=0)
        return corr_h
    
    def forward(self, batch, clf=True, lm=True):
        "Take in and process masked src and target sequences."
        ret = []
        u_h = self.encode(batch.text, batch.text_lengths)
        if clf:
            if self.config['pick_hid']: u = self.pick_h(u_h, batch.text_lengths)
            else: u = u_h[:, -1, :]
            clf_output = self.classifier(u)   
            ret.append(clf_output)   
        if lm:
            text_y = self.generator(u_h)
            ret.append(text_y)
        return ret[0] if len(ret) == 1 else ret
    
    def compute_clf_loss(self, logits, labels):
        loss = self.ce_loss(logits, labels)
        return loss.mean()
    
    def compute_lm_loss(self, text_h, text_y, text_loss_mask):
        loss = self.ce_loss(text_h.contiguous().view(-1, self.config['n_words']), text_y.view(-1)).view(text_h.size(0), -1)
        loss *= text_loss_mask  # mask sequence loss
        return loss.mean()


def make_transformer_model(encoder, config, word_embeddings=None, sememe=False):
    # encoder: dictionary, for vocab
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(config['n_heads'], config['d_model'], config['dpout'])
    ff = PositionwiseFeedForward(config['d_model'], config['d_ff'], config['dpout'])
    position = PositionalEncoding(config) # ctx_embeddings

    embedding_layer = Embeddings(encoder, config, word_embeddings)
    if sememe: 
        sememe_embedding_layer = SememeEmbeddings(config)
        generator_sememe = Generator(config['d_model'], sememe_embedding_layer.lut.weight.shape[0], word_embedding_weight=sememe_embedding_layer.lut.weight)

    if config['tied']:
        if config['train_emb']:
            generator = Generator(config['d_model'], len(encoder), word_embedding_weight=embedding_layer.lut.weight)
        else:
            generator = Generator(config['d_model'], len(encoder), np_word_embedding=word_embeddings)
    else:
        generator = Generator(config['d_model'], len(encoder))



    model = Transformer(
        config,
        Decoder(
            DecoderLayer(config['d_model'], c(attn), c(ff), config['dpout']),
            config['n_layers']),
        nn.Sequential(embedding_layer, c(position)),
        generator,
        sememe_embedding_layer if sememe else None,
        generator_sememe if sememe else None
    )

    for p in model.parameters():
        # we won't update anything that has fixed parameters!
        if p.dim() > 1 and p.requires_grad is True:
            nn.init.xavier_uniform(p)
    return model


def make_lstm_model(encoder, config, word_embeddings=None):
    model = LSTM(config)
    return model
