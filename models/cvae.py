import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from Data import *
from util.misc_utils import move_to_cuda

import numpy as np


class cvae(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(cvae, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.decoder = models.rnn_decoder(config, self.vocab_size, embedding=self.embedding)
        self.config = config
        self.use_content = use_content
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax(-1)
        self.tanh = nn.Tanh()

        self.comment_encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.gama_kld = config.gama_kld

        # prior
        self.prior_fc1_layer = nn.Linear(config.decoder_hidden_size, config.n_z * 2)
        self.prior_mulogvar_layer = nn.Linear(config.n_z * 2, config.n_z * 2)

        # recog
        self.recog_mulogvar_layer = nn.Linear(config.decoder_hidden_size * 2, config.n_z * 2)

        self.z_to_hidden = nn.Linear(config.n_z + config.decoder_hidden_size, config.decoder_hidden_size)

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
        src, src_len, src_mask = batch.title_content, batch.title_content_len, batch.title_content_mask
        contexts, state = self.encoder(src, src_len)

        # prior
        news_rep = state[0][-1]
        prior_mulogvar = self.prior_mulogvar_layer(torch.tanh(self.prior_fc1_layer(news_rep)))
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, dim=1)

        if not is_test:
            # comment encoder
            tgt, tgt_len = batch.tgt, batch.tgt_len
            _, comment_state = self.comment_encoder(tgt, tgt_len)  # output: bsz * n_hidden
            comment_rep = comment_state[0][-1]  # bsz * n_hidden

            # comment vae
            recog_mulogvar = self.recog_mulogvar_layer(torch.cat([news_rep, comment_rep], dim=-1))
            mu, logvar = torch.chunk(recog_mulogvar, 2, dim=1)

            z = torch.randn([comment_rep.size(0), self.config.n_z]).to(mu.device)  # Noise sampled from Normal(0,1)
            z = mu + z * torch.exp(0.5 * logvar)  # Reparameterization trick

            # for loss
            def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
                kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                                       - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                                       - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
                return kld
            kld = gaussian_kld(mu, logvar, prior_mu, prior_logvar)
        else:
            z = prior_mu
            kld = 0.0
        z = self.z_to_hidden(torch.cat([z, news_rep], dim=-1))
        zz = z.unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        state = (zz, zz)
        return contexts, state, kld

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, kld = self.encode(batch)
        tgt = batch.tgt

        outputs, final_state, attns = self.decoder(tgt[:, :-1], state, contexts)
        return {
            'outputs': outputs,
            'kld': kld.mean(),
        }

    def sample(self, batch, use_cuda):
        contexts, state = self.encode(batch, use_cuda)

        bos = torch.ones(contexts.size(0)).long().fill_(self.vocab.word2id('[START]'))
        bos = bos.to(contexts.device)
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts)

        return sample_ids, final_outputs[1]

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1, n_best=1):
        # (1) Run the encoder on the src. Done!!!!
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, enc_state, _ = self.encode(batch, True)

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
