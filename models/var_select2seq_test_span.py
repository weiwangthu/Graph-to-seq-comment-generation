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
        self.linear1 = nn.Linear(config.encoder_hidden_size * 4, config.encoder_hidden_size)
        self.linear2 = nn.Linear(config.encoder_hidden_size, 2)

    def forward(self, contexts, title):
        title = title.unsqueeze(1).expand(-1, contexts.size(1), -1)
        gates = F.softmax(self.linear2(F.relu((self.linear1(torch.cat([contexts, title], dim=-1))))), dim=-1)
        return gates


class PostSelectGate(nn.Module):

    def __init__(self, config):
        super(PostSelectGate, self).__init__()
        self.linear1 = nn.Linear(config.encoder_hidden_size * 6, config.encoder_hidden_size)
        self.linear2 = nn.Linear(config.encoder_hidden_size, 2)

    def forward(self, contexts, title, comment_context):
        title = title.unsqueeze(1).expand(-1, contexts.size(1), -1)
        comment_context = comment_context.unsqueeze(1).expand(-1, contexts.size(1), -1)
        gates = F.softmax(self.linear2(F.relu(self.linear1(torch.cat([contexts, title, comment_context], dim=-1)))), dim=-1)
        return gates


class var_select2seq_test_span(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(var_select2seq_test_span, self).__init__()
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

        self.title_encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        # vae bow and cluster
        # encoder
        self.enc_linear1 = nn.Linear(self.vocab_size, 2 * config.decoder_hidden_size)
        self.enc_linear2 = nn.Linear(2 * config.decoder_hidden_size, config.decoder_hidden_size)

        # select gate
        self.select_gate = SelectGate(config)
        self.select_post_gate = PostSelectGate(config)
        self.gama_kld_select = config.gama_kld_select

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs'].transpose(0, 1)
        word_loss = models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

        # gate loss
        gate_loss = out_dict['l1_gates']

        # kld select
        kld_select = out_dict['kld_select']
        if self.config.min_select > 0:
            kld_select = torch.abs(kld_select - self.config.min_select)

        loss = word_loss[0] + self.config.gama1 * gate_loss + self.gama_kld_select * kld_select
        return {
            'loss': loss,
            'word_loss': word_loss[0],
            'acc': word_loss[1],
            'gate_loss': gate_loss,
            'kld_select_loss': kld_select,
            'pri_gates': out_dict['pri_gates'],
        }

    def merge_local_context(self, group_vectors, group_masks):
        g_vs = []
        g_mask = []
        for i in range(len(group_vectors)):
            g_v = (group_vectors[i] * group_masks[i].unsqueeze(dim=2).float()).sum(dim=1)
            g_len = group_masks[i].float().sum(dim=-1)
            g_mask.append(g_len > 0)

            g_len[g_len == 0] = 1
            g_vs.append(g_v / g_len.unsqueeze(dim=-1))
        g_vs = torch.stack(g_vs, dim=1)
        g_mask = torch.stack(g_mask, dim=1)
        return g_vs, g_mask

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
        local_mask = torch.split(content_mask, self.config.content_span, dim=1)
        local_vectors, local_vector_mask = self.merge_local_context(local_contexts, local_mask)

        # select important information of body
        org_context_gates = self.select_gate(local_vectors, title_rep)  # output: bsz * n_context * 2
        context_gates = gumbel_softmax(torch.log(org_context_gates + 1e-10), self.config.tau)
        context_gates = context_gates[:, :, 0]  # bsz * n_context
        org_context_gates = org_context_gates[:, :, 0]  # bsz * n_context

        if not is_test:
            # comment encoder
            tgt_bow = batch.tgt_bow
            enc_hidden = torch.tanh(self.enc_linear1(tgt_bow.float()))
            enc_hidden = torch.tanh(self.enc_linear2(enc_hidden))

            # selector vae
            comment_rep = enc_hidden
            org_post_context_gates = self.select_post_gate(local_vectors, title_rep, comment_rep)
            post_context_gates = gumbel_softmax(torch.log(org_post_context_gates + 1e-10), self.config.tau)
            post_context_gates = post_context_gates[:, :, 0]  # bsz * n_context
            org_post_context_gates = org_post_context_gates[:, :, 0]

            # kl(p1||p2)
            def kldiv(p1, p2):
                kl = p1 * torch.log((p1 + 1e-10) / (p2 + 1e-10)) + (1 - p1) * torch.log((1 - p1 + 1e-10) / (1 - p2 + 1e-10))
                return kl

            kld_select = ((kldiv(org_post_context_gates, org_context_gates) * local_vector_mask.float()).sum(dim=-1) / local_vector_mask.float().sum(dim=-1)).mean()


        else:
            comment_rep = None
            kld_select = 0.0

            # random
            # context_gates = torch.bernoulli(context_gates)

            # gumbel
            # context_gates = gumbel_softmax(torch.log(org_context_gates + 1e-10), self.config.tau)
            # context_gates = context_gates[:, :, 0]

            # best
            org_context_gates[org_context_gates > self.config.gate_prob] = 1.0
            org_context_gates[org_context_gates <= self.config.gate_prob] = 0.0

            post_context_gates = org_context_gates
            context_gates = org_context_gates

        l1_gates = (post_context_gates * local_vector_mask.float()).sum(dim=-1) / local_vector_mask.float().sum(dim=-1)
        pri_gates = (context_gates * local_vector_mask.float()).sum(dim=-1) / local_vector_mask.float().sum(dim=-1)

        context_gates = context_gates.unsqueeze(dim=-1).expand(-1, -1, self.config.content_span)
        context_gates = context_gates.reshape(context_gates.size(0), -1)[:, :contexts.size(1)]
        post_context_gates = post_context_gates.unsqueeze(dim=-1).expand(-1, -1, self.config.content_span)
        post_context_gates = post_context_gates.reshape(post_context_gates.size(0), -1)[:, :contexts.size(1)]

        # collect title and body
        one_gates = torch.ones_like(title_contexts[:, :, 0])
        all_contexts = torch.cat([title_contexts, contexts], dim=1)
        all_post_context_gates = torch.cat([one_gates, post_context_gates], dim=1)
        all_context_gates = torch.cat([one_gates, context_gates], dim=1)

        return all_contexts, state, all_post_context_gates, comment_rep, kld_select, all_context_gates, l1_gates, pri_gates

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, post_context_gates, comment_rep, kld_select, context_gates, l1_gates, pri_gates = self.encode(batch)

        tgt, tgt_len = batch.tgt, batch.tgt_len

        if self.config.use_post_gate:
            gates = post_context_gates
        else:
            gates = context_gates
        # decoder
        outputs, final_state, attns = self.decoder(tgt[:, :-1], state, contexts, gates)

        return {
            'outputs': outputs,
            'l1_gates': l1_gates.mean(),
            'kld_select': kld_select,
            'pri_gates': pri_gates.mean(),
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
        contexts, enc_state, org_context_gates, _, _, _ = self.encode(batch, True)

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
        context_gates = org_context_gates.repeat(beam_size, 1)
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
        if self.config.debug_select:
            title_content = batch.title_content
            title_content[org_context_gates == 0] = self.vocab.PAD_token
            all_select_words = title_content
            return allHyps, allAttn, all_select_words
        else:
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