import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from Data import *
from util.misc_utils import move_to_cuda

import numpy as np


class GetUser(nn.Module):

    def __init__(self, config):
        super(GetUser, self).__init__()
        self.linear = nn.Linear(config.decoder_hidden_size, config.n_topic_num)
        self.use_emb = nn.Embedding(config.n_topic_num, config.n_z)
        self.topic_id = -1
        self.config = config

    def forward(self, latent_context, is_test=False):
        if not is_test:
            p_user = F.softmax(self.linear(latent_context), dim=-1)  # bsz * 10

            if self.config.one_user:
                p_user = gumbel_softmax(torch.log(p_user + 1e-10), self.config.tau)

            h_user = (self.use_emb.weight.unsqueeze(0) * p_user.unsqueeze(-1)).sum(dim=1)  # bsz * n_hidden
            selected_user = torch.argmax(p_user, dim=-1)
        else:
            if self.topic_id == -1:
                ids = torch.LongTensor(latent_context.size(0)).to(latent_context.device).random_(0, self.use_emb.weight.size(0))
            else:
                ids = torch.LongTensor(latent_context.size(0)).to(latent_context.device).fill_(self.topic_id)
            h_user = self.use_emb(ids)
            selected_user = ids
            p_user = None
        return h_user, selected_user, p_user

class user_autoenc(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(user_autoenc, self).__init__()
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
        if config.n_z != config.decoder_hidden_size:
            self.mu_to_hidden = nn.Linear(config.n_z, config.decoder_hidden_size)
        else:
            self.mu_to_hidden = lambda x: x

        self.get_user = GetUser(config)

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs'].transpose(0, 1)
        word_loss = models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

        # rank and reg loss
        reg_loss = out_dict['reg']
        dis_loss = out_dict['dis']

        p_user = out_dict['p_user']

        avg_batch_select_prob = p_user.mean(dim=0)
        select_entropy = avg_batch_select_prob * torch.log(avg_batch_select_prob + 1e-20)
        select_entropy = select_entropy.mean()

        loss = word_loss[0] + self.config.gama_reg * reg_loss + self.config.gama_select * select_entropy \
        + self.config.gama_rank * dis_loss

        return {
            'loss': loss,
            'word_loss': word_loss[0],
            'acc': word_loss[1],
            'reg': reg_loss,
            'user_norm': out_dict['user_norm'],
            'selected_user': out_dict['selected_user'],
            'select_entropy': select_entropy,
            'dis': dis_loss
        }

    def encode(self, batch, is_test=False):
        tgt, tgt_len = batch.tgt, batch.tgt_len
        _, comment_state = self.comment_encoder(tgt, tgt_len)  # output: bsz * n_hidden
        comment_rep = comment_state[0][-1]  # bsz * n_hidden
        return comment_rep

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        comment_rep = self.encode(batch)

        tgt, tgt_len = batch.tgt, batch.tgt_len

        # get user
        h_user, selected_user, p_user = self.get_user(comment_rep)
        h_user = self.mu_to_hidden(h_user)

        # user loss
        reg_loss = torch.mm(self.get_user.use_emb.weight, self.get_user.use_emb.weight.t()) - torch.eye(self.get_user.use_emb.weight.size(0), dtype=h_user.dtype, device=h_user.device)
        reg_loss = torch.norm(reg_loss)

        # distance loss
        mse_e = ((comment_rep.detach() - h_user) ** 2).mean()
        mse_z = ((h_user.detach() - comment_rep) ** 2).mean()
        dis_loss = mse_e + 0.25 * mse_z

        # decoder
        zz = h_user.unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        dec_state = (zz, zz)
        outputs, final_state, attns = self.decoder(tgt[:, :-1], dec_state, None, None)
        # return outputs, gates, title_state[0], comment_state[0]

        user_norm = torch.norm(self.get_user.use_emb.weight, 2, dim=1).mean()
        return {
            'outputs': outputs,
            'comment_state': comment_rep,
            'reg': reg_loss,
            'user_norm': user_norm,
            'selected_user': selected_user,
            'p_user': p_user,
            'dis': dis_loss,
        }

    def sample(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, comment_rep = self.encode(batch, True)
        h_user, _ = self.get_user(contexts, True)

        bos = torch.ones(contexts.size(0)).long().fill_(self.vocab.word2id('[START]'))
        bos = bos.to(contexts.device)
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts, None, h_user)

        return sample_ids, final_outputs[1]

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1, n_best=1):
        # (1) Run the encoder on the src. Done!!!!
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, enc_state, comment_rep = self.encode(batch, True)
        h_user, _ = self.get_user(contexts, True)

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
        h_user = h_user.repeat(beam_size, 1)
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
            output, dec_state, attn = self.decoder.sample_one(inp, dec_state, contexts, None, h_user)
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


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(device=logits.device, dtype=logits.dtype)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    return (y_hard - y).detach() + y