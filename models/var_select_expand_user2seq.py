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
        self.linear1 = nn.Linear(config.encoder_hidden_size * 2, config.encoder_hidden_size)
        self.linear2 = nn.Linear(config.encoder_hidden_size, 2)

    def forward(self, contexts):
        gates = F.softmax(self.linear2(F.relu((self.linear1(contexts)))), dim=-1)
        return gates


class PostSelectGate(nn.Module):

    def __init__(self, config):
        super(PostSelectGate, self).__init__()
        self.linear1 = nn.Linear(config.encoder_hidden_size * 4, config.encoder_hidden_size)
        self.linear2 = nn.Linear(config.encoder_hidden_size, 2)

    def forward(self, contexts, comment_context):
        comment_context = comment_context.unsqueeze(1).expand(-1, contexts.size(1), -1)
        gates = F.softmax(self.linear2(F.relu(self.linear1(torch.cat([contexts, comment_context], dim=-1)))), dim=-1)
        return gates


class LatentMap(nn.Module):

    def __init__(self, config):
        super(LatentMap, self).__init__()
        self.n_layers = config.n_topic_num
        self.latents = nn.ModuleList([])
        for i in range(self.n_layers):
            self.latents.append(nn.Linear(config.decoder_hidden_size, config.n_z))

    def forward(self, title_context):
        topics = []
        for i in range(self.n_layers):
            topics.append(self.latents[i](title_context))
        topics = torch.stack(topics, 1)
        return topics


class GetUser:

    def __init__(self, config):
        self.topic_id = -1
        self.config = config

    def forward(self, topics):
        if self.topic_id == -1:
            ids = torch.LongTensor(topics.size(0), 1, topics.size(-1)).to(topics.device).random_(0, topics.size(-1))
        else:
            ids = torch.LongTensor(topics.size(0), 1, topics.size(-1)).to(topics.device).fill_(self.topic_id)
        h_user = topics.gather(dim=1, index=ids).squeeze(dim=1)
        selected_user = ids
        return h_user, selected_user

class var_select_expand_user2seq(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(var_select_expand_user2seq, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.decoder = models.rnn_topic_decoder(config, self.vocab_size, embedding=self.embedding)
        self.config = config
        self.use_content = use_content
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax(-1)
        self.tanh = nn.Tanh()

        # select gate
        self.comment_encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.map_to_latent = LatentMap(config)

        self.get_user = GetUser(config)

        self.select_gate = SelectGate(config)
        self.select_post_gate = PostSelectGate(config)
        self.gama_kld_select = config.gama_kld_select

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs'].transpose(0, 1)
        word_loss = models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

        # rank and reg loss
        rank_loss = out_dict['rank']

        loss = word_loss[0] + self.config.gama_rank * rank_loss

        # gate loss
        gate_loss = out_dict['l1_gates']

        # kld select
        kld_select = out_dict['kld_select']
        if self.config.min_select > 0:
            kld_select = torch.abs(kld_select - self.config.min_select)

        loss += self.config.gama1 * gate_loss + self.gama_kld_select * kld_select
        return {
            'loss': loss,
            'word_loss': word_loss[0],
            'acc': word_loss[1],
            'selected_user': out_dict['selected_user'],
            'rank': rank_loss,
            'gate_loss': gate_loss,
            'kld_select_loss': kld_select,
            'pri_gates': out_dict['pri_gates'],
        }

    def encode(self, batch, is_test=False):
        # content, content_len, content_mask = batch.content, batch.cotent_len, batch.cotent_mask
        content, content_len, content_mask = batch.title_content, batch.title_content_len, batch.title_content_mask

        # encoder
        contexts, state = self.encoder(content, content_len)

        # select important information of body
        org_context_gates = self.select_gate(contexts)  # output: bsz * n_context * 2
        context_gates = org_context_gates[:, :, 0]  # bsz * n_context

        if not is_test:
            # comment encoder
            tgt, tgt_len = batch.tgt, batch.tgt_len
            _, comment_state = self.comment_encoder(tgt, tgt_len)  # output: bsz * n_hidden
            comment_rep = comment_state[0][-1]  # bsz * n_hidden

            # selector vae
            org_post_context_gates = self.select_post_gate(contexts, comment_rep)
            post_context_gates = gumbel_softmax(torch.log(org_post_context_gates + 1e-10), self.config.tau)
            post_context_gates = post_context_gates[:, :, 0]  # bsz * n_context
            org_post_context_gates = org_post_context_gates[:, :, 0]

            # kl(p1||p2)
            def kldiv(p1, p2):
                kl = p1 * torch.log((p1 + 1e-10) / (p2 + 1e-10)) + (1 - p1) * torch.log((1 - p1 + 1e-10) / (1 - p2 + 1e-10))
                return kl

            kld_select = ((kldiv(org_post_context_gates, context_gates) * content_mask.float()).sum(dim=-1) / content_len.float()).mean()

        else:
            comment_rep = None
            kld_select = 0.0

            # random
            # context_gates = torch.bernoulli(context_gates)

            # gumbel
            # context_gates = gumbel_softmax(torch.log(org_context_gates + 1e-10), self.config.tau)
            # context_gates = context_gates[:, :, 0]

            # best
            context_gates[context_gates > self.config.gate_prob] = 1.0
            context_gates[context_gates <= self.config.gate_prob] = 0.0

            post_context_gates = context_gates

        return contexts, state, post_context_gates, comment_rep, kld_select, context_gates

    def topic_attention(self, comment, topics):
        comment = comment.unsqueeze(2)  # bsz * n_hidden * 1
        weights = torch.bmm(topics, comment).squeeze(2)  # bsz * n_topic
        weights = F.softmax(weights, dim=-1)
        return weights

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, post_context_gates, comment_rep, kld_select, context_gates = self.encode(batch)

        content_len, content_mask = batch.title_content_len, batch.title_content_mask
        tgt, tgt_len = batch.tgt, batch.tgt_len

        # map to multi topic
        gate_mask = (post_context_gates > 0.5) & content_mask
        gate_len = gate_mask.float().sum(dim=-1) + 1
        init_state = (contexts * gate_mask.float().unsqueeze(dim=2)).sum(dim=1) / gate_len.unsqueeze(dim=1)
        topics = self.map_to_latent(init_state)  # output: bsz * n_topic * n_hidden
        topic_gates = self.topic_attention(comment_rep, topics)  # output: bsz * n_topic
        topic_gates = gumbel_softmax(torch.log(topic_gates), self.config.tau)  # bsz * n_topic
        selected_user = torch.argmax(topic_gates, dim=-1)

        select_topic = (topics * topic_gates.unsqueeze(-1)).sum(dim=1)  # bsz * n_hidden

        # decoder
        outputs, final_state, attns = self.decoder(tgt[:, :-1], state, contexts, post_context_gates, select_topic)

        # match loss
        news_rep = state[0][-1]
        news_rep_neg = torch.roll(news_rep, 1, dims=0)

        # user loss
        rank_loss = (1 - torch.sum(comment_rep * news_rep, dim=-1) + torch.sum(comment_rep * news_rep_neg, dim=-1)).clamp(min=0).mean()

        l1_gates = (post_context_gates * content_mask.float()).sum(dim=-1) / content_len.float()
        pri_gates = (context_gates * content_mask.float()).sum(dim=-1) / content_len.float()
        return {
            'outputs': outputs,
            'selected_user': selected_user,
            'rank': rank_loss,
            'l1_gates': l1_gates.mean(),
            'kld_select': kld_select,
            'pri_gates': pri_gates.mean(),
        }

    def sample(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, context_gates, comment_rep, _, _ = self.encode(batch, True)
        h_user, _ = self.get_user(contexts, True)

        bos = torch.ones(contexts.size(0)).long().fill_(self.vocab.word2id('[START]'))
        bos = bos.to(contexts.device)
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts, context_gates, h_user)

        return sample_ids, final_outputs[1]

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1, n_best=1):
        # (1) Run the encoder on the src. Done!!!!
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, enc_state, context_gates, comment_rep, _, _ = self.encode(batch, True)
        content_len, content_mask = batch.title_content_len, batch.title_content_mask

        # map to multi topic
        gate_mask = (context_gates > 0.5) & content_mask
        gate_len = gate_mask.float().sum(dim=-1) + 1
        init_state = (contexts * gate_mask.float().unsqueeze(dim=2)).sum(dim=1) / gate_len.unsqueeze(dim=1)
        topics = self.map_to_latent(init_state)  # output: bsz * n_topic * n_hidden
        h_user, _ = self.get_user.forward(topics)

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
            output, dec_state, attn = self.decoder.sample_one(inp, dec_state, contexts, context_gates, h_user)
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