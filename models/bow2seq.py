import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from torch.nn.utils.rnn import pad_sequence
from Data import *

import numpy as np


class bow2seq(nn.Module):
    def __init__(self, config, vocab, use_cuda, pretrain=None):
        super(bow2seq, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        # self.encoder = models.GCN_Encoder(config, self.vocab_size, embedding=self.embedding)
        self.bert_encoder = models.bert.BERT(config.head_num, config.emb_size, config.dropout,
                                             config.emb_size, self.vocab_size, config.num_layers,
                                             config.max_article_len, word_emb=self.embedding)
        self.decoder = models.rnn_decoder(config, self.vocab_size, embedding=self.embedding)
        self.proj = nn.Linear(config.emb_size, config.decoder_hidden_size)
        self.state_wc = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size * config.num_layers)
        self.state_wh = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size * config.num_layers)
        self.tanh = nn.Tanh()
        self.config = config
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax(-1)

    def compute_loss(self, hidden_outputs, targets):
        assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
        targets = targets.contiguous().view(-1)
        weight = torch.ones(outputs.size(-1))
        weight[PAD] = 0
        weight[UNK] = 0
        weight = weight.to(outputs.device)
        loss = F.nll_loss(torch.log(outputs), targets, weight=weight, reduction='sum')
        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets.data).masked_select(targets.ne(PAD).data).sum()
        num_total = targets.ne(PAD).data.sum()
        loss = loss.div(num_total.float())
        acc = num_correct.float() / num_total.float()
        return loss, acc

    def build_init_state(self, state, num_layers):
        c0 = self.tanh(self.state_wc(state)).contiguous().view(-1, num_layers, self.config.decoder_hidden_size)
        h0 = self.tanh(self.state_wh(state)).contiguous().view(-1, num_layers, self.config.decoder_hidden_size)
        c0 = c0.transpose(1, 0)
        h0 = h0.transpose(1, 0)
        return c0, h0

    def forward(self, batch, use_cuda):
        src, src_mask = batch.bow, batch.bow_mask
        tgt, tgt_len, tgt_mask = batch.tgt, batch.tgt_len, batch.tgt_mask
        if use_cuda:
            src, src_mask = src.cuda(), src_mask.cuda()
            tgt = tgt.cuda()
        contexts = self.bert_encoder(src, src_mask)
        contexts = self.proj(contexts)
        c0, h0 = self.build_init_state(contexts[:, 0], self.config.num_layers)
        outputs, final_state, attns = self.decoder(tgt[:, :-1], (c0, h0), contexts)
        outputs = F.softmax(outputs, -1)
        return outputs

    def sample(self, batch, use_cuda):
        src, src_mask = batch.bow, batch.bow_mask
        if use_cuda:
            src, src_mask = src.cuda(), src_mask.cuda()
        contexts = self.bert_encoder(src, src_mask)
        contexts = self.proj(contexts)
        c0, h0 = self.build_init_state(contexts[:, 0], self.config.num_layers)
        state = (c0, h0)

        bos = torch.ones(contexts.size(0)).long().fill_(self.vocab.word2id('[START]'))
        bos = bos.to(contexts.device)
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts)

        return sample_ids, final_outputs[1]

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1, n_best=1):
        # (1) Run the encoder on the src. Done!!!!
        src, src_mask = batch.bow, batch.bow_mask
        if use_cuda:
            src, src_mask = src.cuda(), src_mask.cuda()
        contexts = self.bert_encoder(src, src_mask)
        contexts = self.proj(contexts)
        c0, h0 = self.build_init_state(contexts[:, 0], self.config.num_layers)
        enc_state = (c0, h0)

        batch_size = contexts.size(0)
        beam = [models.Beam(beam_size, n_best=1, cuda=use_cuda)
                for _ in range(batch_size)]

        def rvar(a):
            return a.repeat(1, beam_size, 1)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = contexts.repeat(beam_size, 1, 1)
        dec_state = (rvar(enc_state[0]), rvar(enc_state[1]))  # layer, beam*batch, nh

        # (2) run the decoder to generate sentences, using beam search.
        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break

            # Construct beam*batch  nxt words.
            # Get all the pending current beam words and arrange for forward.
            # beam is batch_sized, so stack on dimension 1 not 0
            inp = torch.stack([b.getCurrentState() for b in beam], 1).contiguous().view(-1)
            if use_cuda:
                inp = inp.cuda()

            # Run one step.
            output, dec_state, attn = self.decoder.sample_one(inp, dec_state, contexts)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):  # there are batch size beams!!! so here enumerate over batch
                b.advance(output.data[:, j], attn.data[:, j])  # output is beam first
                b.beam_update(dec_state, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []

        for j in range(batch_size):
            b = beam[j]
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps)
            allScores.append(scores)
            allAttn.append(attn)

        # print(allHyps)
        # print(allAttn)
        return allHyps, allAttn
