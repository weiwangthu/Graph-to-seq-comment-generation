import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
from torch.nn.utils.rnn import pad_sequence
from Data import *

import numpy as np


class attentive_pooling(nn.Module):
    def __init__(self, hidden_size):
        super(attentive_pooling, self).__init__()
        self.w = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, memory, mask=None):
        h = torch.tanh(self.w(memory))
        score = torch.squeeze(self.u(h), -1)
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, 1)
        return s


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirec):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout, bidirectional=bidirec,
                           batch_first=True)

    def forward(self, input, lengths):
        length, indices = torch.sort(lengths, dim=0, descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = list(torch.unbind(length, dim=0))
        embs = pack(torch.index_select(input, dim=0, index=indices), input_length, batch_first=True)
        outputs, _ = self.rnn(embs)
        outputs = unpack(outputs, batch_first=True)[0]
        outputs = torch.index_select(outputs, dim=0, index=ind)
        return outputs


class hierarchical_attention(nn.Module):
    def __init__(self, config, vocab, use_cuda, pretrain=None):
        super(hierarchical_attention, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.word_encoder = Encoder(config.emb_size, config.encoder_hidden_size, config.num_layers, config.dropout, config.bidirec)
        self.word_attentive_pool = attentive_pooling(config.encoder_hidden_size * 2)
        self.sentence_encoder = Encoder(config.encoder_hidden_size * 2, config.encoder_hidden_size * 2, config.num_layers,
                                        config.dropout, config.bidirec)
        self.sentence_attentive_pool = attentive_pooling(config.decoder_hidden_size)
        self.decoder = models.rnn_decoder(config, self.vocab_size, embedding=self.embedding)
        self.w_context = nn.Linear(config.decoder_hidden_size * 2, config.decoder_hidden_size, bias=False)
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

    def encode(self, contents, contents_mask, contents_length, sent_mask):
        sent_vec_batch = []
        for content, content_mask, content_length in zip(contents, contents_mask, contents_length):
            length = torch.sum(content_mask, -1)
            emb = self.embedding(content)
            context = self.word_encoder(emb, length)
            sent_vec = self.word_attentive_pool(context, content_mask)
            sent_vec_batch.append(sent_vec)
            assert len(sent_vec) == content_length, (len(sent_vec), content_length)  # sentence number
        sent_vec_batch = pad_sequence(sent_vec_batch, batch_first=True)
        sent_hidden = self.sentence_encoder(sent_vec_batch, contents_length)
        sent_hidden = self.w_context(sent_hidden)
        state = self.sentence_attentive_pool(sent_hidden, sent_mask)
        return sent_hidden, state

    def build_init_state(self, state, num_layers):
        c0 = self.tanh(self.state_wc(state)).contiguous().view(-1, num_layers, self.config.decoder_hidden_size)
        h0 = self.tanh(self.state_wh(state)).contiguous().view(-1, num_layers, self.config.decoder_hidden_size)
        c0 = c0.transpose(1, 0)
        h0 = h0.transpose(1, 0)
        return c0, h0

    def forward(self, batch, use_cuda):
        src, src_mask, src_len = batch.sentence_content, batch.sentence_content_mask, batch.sentence_content_len
        sent_mask = batch.sentence_mask
        tgt, tgt_len, tgt_mask = batch.tgt, batch.tgt_len, batch.tgt_mask
        if use_cuda:
            tgt = tgt.cuda()
            src = [s.cuda() for s in src]
            src_mask = [s.cuda() for s in src_mask]
            src_len = src_len.cuda()
            sent_mask = sent_mask.cuda()
        context, state = self.encode(src, src_mask, src_len, sent_mask)
        c0, h0 = self.build_init_state(state, self.config.num_layers)
        outputs, final_state, _ = self.decoder(tgt[:, :-1], (c0, h0), context)
        outputs = F.softmax(outputs, -1)
        return outputs

    def sample(self, batch, use_cuda):
        src, src_mask, src_len = batch.sentence_content, batch.sentence_content_mask, batch.sentence_content_len
        sent_mask = batch.sentence_mask
        bos = torch.ones(len(src)).long().fill_(self.vocab.word2id('[START]'))
        if use_cuda:
            src = [s.cuda() for s in src]
            src_mask = [s.cuda() for s in src_mask]
            src_len = src_len.cuda()
            sent_mask = sent_mask.cuda()
            bos = bos.cuda()
        context, state = self.encode(src, src_mask, src_len, sent_mask)
        c0, h0 = self.build_init_state(state, self.config.num_layers)
        sample_ids, final_outputs = self.decoder.sample([bos], (c0, h0), context)

        return sample_ids, None

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1, n_best=1):
        # (1) Run the encoder on the src. Done!!!!
        src, src_mask, src_len = batch.sentence_content, batch.sentence_content_mask, batch.sentence_content_len
        sent_mask = batch.sentence_mask
        if use_cuda:
            src = [s.cuda() for s in src]
            src_mask = [s.cuda() for s in src_mask]
            src_len = src_len.cuda()
            sent_mask = sent_mask.cuda()
        contexts, enc_state = self.encode(src, src_mask, src_len, sent_mask)
        enc_state = self.build_init_state(enc_state, self.config.num_layers)

        batch_size = contexts.size(0)
        beam = [models.Beam(beam_size, n_best=1, cuda=use_cuda)
                for _ in range(batch_size)]

        #  (1b) Initialize for the decoder.
        def rvar(a):
            return a.repeat(1, beam_size, 1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        # (batch, seq, nh) -> (beam*batch, seq, nh)
        contexts = contexts.repeat(beam_size, 1, 1)
        # (batch, seq) -> (beam*batch, seq)
        # src_mask = src_mask.repeat(beam_size, 1)
        # assert contexts.size(0) == src_mask.size(0), (contexts.size(), src_mask.size())
        # assert contexts.size(1) == src_mask.size(1), (contexts.size(), src_mask.size())
        dec_state = (rvar(enc_state[0]), rvar(enc_state[1]))  # layer, beam*batch, nh
        # decState.repeat_beam_size_times(beam_size)

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
