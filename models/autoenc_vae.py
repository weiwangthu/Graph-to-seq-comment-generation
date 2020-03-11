import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from Data import *
from util.misc_utils import move_to_cuda

import numpy as np


class autoenc_vae(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(autoenc_vae, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.decoder = models.rnn_decoder(config, self.vocab_size, embedding=self.embedding)
        self.config = config
        self.use_content = use_content
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax(-1)
        self.tanh = nn.Tanh()

        self.comment_encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.hidden_to_mu = nn.Linear(2 * config.encoder_hidden_size, config.n_z)
        self.hidden_to_logvar = nn.Linear(2 * config.encoder_hidden_size, config.n_z)
        self.gama_kld = config.gama_kld

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs'].transpose(0, 1)
        word_loss = models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

        # kld comment
        kld = out_dict['kld']

        loss = word_loss[0] + self.gama_kld * kld
        return {
            'loss': loss,
            'word_loss': word_loss[0],
            'acc': word_loss[1],
            'kld_loss': kld,
        }

    def encode(self, batch, is_test=False):
        if not is_test:
            # comment encoder
            tgt, tgt_len = batch.tgt, batch.tgt_len
            _, comment_state = self.comment_encoder(tgt, tgt_len)  # output: bsz * n_hidden
            comment_rep = comment_state[0][-1]  # bsz * n_hidden

            # comment vae
            mu = self.hidden_to_mu(comment_rep)  # Get mean of lantent z
            logvar = self.hidden_to_logvar(comment_rep)  # Get log variance of latent z

            z = torch.randn([comment_rep.size(0), self.config.n_z]).to(mu.device)  # Noise sampled from Normal(0,1)
            z = mu + z * torch.exp(0.5 * logvar)  # Reparameterization trick
            kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()  # Compute KL divergence loss
        else:
            z = torch.randn([batch.title_content.size(0), self.config.n_z]).to(batch.title_content.device)
            kld = 0.0

        return z, kld

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        z, kld = self.encode(batch)

        tgt, tgt_len = batch.tgt, batch.tgt_len

        # decoder
        zz = z.unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        dec_state = (zz, zz)
        outputs, final_state, attns = self.decoder(tgt[:, :-1], dec_state, None, None)
        # return outputs, gates, title_state[0], comment_state[0]

        return {
            'outputs': outputs,
            'kld': kld,
        }

    def get_comment_rep(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        # comment encoder
        tgt, tgt_len = batch.tgt, batch.tgt_len
        _, comment_state = self.comment_encoder(tgt, tgt_len)  # output: bsz * n_hidden
        comment_rep = comment_state[0][-1]  # bsz * n_hidden
        # comment_rep = self.hidden_to_mu(comment_rep)  # Get mean of lantent z
        return comment_rep

    def sample(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        z, kld = self.encode(batch, True)
        zz = z.unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        dec_state = (zz, zz)

        bos = torch.ones(z.size(0)).long().fill_(self.vocab.word2id('[START]'))
        bos = bos.to(z.device)
        sample_ids, final_outputs = self.decoder.sample([bos], dec_state, None, None)

        return sample_ids, final_outputs[1]

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1, n_best=1):
        # (1) Run the encoder on the src. Done!!!!
        if use_cuda:
            batch = move_to_cuda(batch)
        z, kld = self.encode(batch, True)
        zz = z.unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        dec_state = (zz, zz)

        batch_size = z.size(0)
        beam = [models.Beam(beam_size, n_best=1, cuda=use_cuda)
                for _ in range(batch_size)]

        #  (1b) Initialize for the decoder.
        def rvar(a):
            return a.repeat(1, beam_size, 1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        # (batch, seq, nh) -> (beam*batch, seq, nh)
        # (batch, seq) -> (beam*batch, seq)
        # src_mask = src_mask.repeat(beam_size, 1)
        # assert contexts.size(0) == src_mask.size(0), (contexts.size(), src_mask.size())
        # assert contexts.size(1) == src_mask.size(1), (contexts.size(), src_mask.size())
        dec_state = (rvar(dec_state[0]), rvar(dec_state[1]))  # layer, beam*batch, nh
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
            output, dec_state, attn = self.decoder.sample_one(inp, dec_state, None, None)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            if attn is None:
                attn = output
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
