import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from Data import *
from util.misc_utils import move_to_cuda

import numpy as np


class SelectGate(nn.Module):

    def __init__(self, config):
        super(SelectGate, self).__init__()
        self.linear1 = nn.Linear(config.encoder_hidden_size * 2, config.encoder_hidden_size * 2)

    def forward(self, context):
        converted_context = F.tanh((self.linear1(context)))
        return converted_context


class var_select2seq_align(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(var_select2seq_align, self).__init__()
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

        # select gate
        self.title_encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.comment_encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)

        self.select_gate = SelectGate(config)
        self.gama_kld_select = config.gama_kld_select

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs'].transpose(0, 1)
        word_loss = models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

        # kld select
        kld_select = out_dict['kld_select']
        if self.config.min_select > 0:
            kld_select = torch.abs(kld_select - self.config.min_select)

        loss = word_loss[0] + self.gama_kld_select * kld_select

        # rank and reg loss
        rank_loss = out_dict['rank']
        loss += self.config.gama_rank * rank_loss
        return {
            'loss': loss,
            'word_loss': word_loss[0],
            'acc': word_loss[1],
            'kld_select_loss': kld_select,
            'rank': rank_loss,
        }

    def topic_attention(self, comment, topics):
        comment = comment.unsqueeze(2)  # bsz * n_hidden * 1
        weights = torch.bmm(topics, comment).squeeze(2)  # bsz * n_topic
        weights = F.softmax(weights, dim=-1)
        return weights

    def merge_local_context(self, group_vectors):
        g_v = []
        for g in group_vectors:
            g_v.append(g.max(dim=1)[0])
        g_v = torch.stack(g_v, dim=1)
        return g_v

    def encode(self, batch, is_test=False):
        src, src_len, src_mask = batch.title, batch.title_len, batch.title_mask
        content, content_len, content_mask = batch.content, batch.content_len, batch.content_mask
        # content, content_len, content_mask = batch.title_content, batch.title_content_len, batch.title_content_mask

        # input: title, content
        title_contexts, title_state = self.title_encoder(src, src_len)
        title_rep = title_state[0][-1]  # bsz * n_hidden

        # encoder
        contexts, state = self.encoder(content, content_len)
        local_contexts = torch.split(contexts, self.config.content_span, dim=1)
        local_vectors = self.merge_local_context(local_contexts)

        # new_rep = state[0][-1]
        converted_new_rep = self.select_gate(title_rep)
        org_local_scores = self.topic_attention(converted_new_rep, local_vectors)
        local_scores = gumbel_softmax(torch.log(org_local_scores), self.config.tau)  # bsz * n_topic
        context_gates = local_scores.unsqueeze(dim=-1).expand(-1, -1, self.config.content_span)
        context_gates = context_gates.reshape(context_gates.size(0), -1)[:, :contexts.size(1)]

        if not is_test:
            # comment encoder
            tgt, tgt_len = batch.tgt, batch.tgt_len
            _, comment_state = self.comment_encoder(tgt, tgt_len)  # output: bsz * n_hidden
            comment_rep = comment_state[0][-1]  # bsz * n_hidden

            org_post_local_scores = self.topic_attention(comment_rep, local_vectors)
            post_local_scores = gumbel_softmax(torch.log(org_post_local_scores), self.config.tau)  # bsz * n_topic
            post_context_gates = post_local_scores.unsqueeze(dim=-1).expand(-1, -1, 20)
            post_context_gates = post_context_gates.reshape(post_context_gates.size(0), -1)[:, :contexts.size(1)]

            def kld(p1, p2):
                k = p1 * torch.log((p1 + 1e-20) / (p2 + 1e-20))
                return k.sum(dim=-1)

            kld_select = kld(org_post_local_scores, org_local_scores).mean()
        else:
            comment_rep = None
            post_context_gates = context_gates
            kld_select = 0
        # collect title and body
        one_gates = torch.ones_like(title_contexts[:,:,0])
        all_contexts = torch.cat([title_contexts, contexts], dim=1)
        all_post_context_gates = torch.cat([one_gates, post_context_gates], dim=1)
        all_context_gates = torch.cat([one_gates, context_gates], dim=1)

        return all_contexts, state, all_post_context_gates, comment_rep, kld_select, all_context_gates, converted_new_rep

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, post_context_gates, comment_rep, kld_select, context_gates, converted_new_rep = self.encode(batch)

        tgt, tgt_len = batch.tgt, batch.tgt_len

        if self.config.use_post_gate:
            gates = post_context_gates
        else:
            gates = context_gates
        # decoder
        outputs, final_state, attns = self.decoder(tgt[:, :-1], state, contexts, gates)
        # return outputs, gates, title_state[0], comment_state[0]

        # match loss
        news_rep = converted_new_rep
        news_rep_neg = torch.roll(news_rep, 1, dims=0)

        # user loss
        rank_loss = (1 - torch.sum(comment_rep * news_rep, dim=-1) + torch.sum(comment_rep * news_rep_neg, dim=-1)).clamp(min=0).mean()

        return {
            'outputs': outputs,
            'comment_state': comment_rep,
            'kld_select': kld_select,
            'rank': rank_loss,
        }

    def sample(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, context_gates, _, _, _ = self.encode(batch, True)

        bos = torch.ones(contexts.size(0)).long().fill_(self.vocab.word2id('[START]'))
        bos = bos.to(contexts.device)
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts, context_gates)

        return sample_ids, final_outputs[1]

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1, n_best=1):
        # (1) Run the encoder on the src. Done!!!!
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, enc_state, context_gates, _, _, _ = self.encode(batch, True)

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
        context_gates = context_gates.repeat(beam_size, 1)
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
            output, dec_state, attn = self.decoder.sample_one(inp, dec_state, contexts, context_gates)
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