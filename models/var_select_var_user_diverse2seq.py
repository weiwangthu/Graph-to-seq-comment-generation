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
        self.linear = nn.Linear(config.encoder_hidden_size * 4, 2)

    def forward(self, contexts, title_context):
        title_context = title_context.unsqueeze(1).expand(-1, contexts.size(1), -1)
        gates = F.softmax(self.linear(torch.cat([contexts, title_context], dim=-1)), dim=-1)
        return gates


class PostSelectGate(nn.Module):

    def __init__(self, config):
        super(PostSelectGate, self).__init__()
        self.linear = nn.Linear(config.encoder_hidden_size * 6, 2)

    def forward(self, contexts, title_context, comment_context):
        title_context = title_context.unsqueeze(1).expand(-1, contexts.size(1), -1)
        comment_context = comment_context.unsqueeze(1).expand(-1, contexts.size(1), -1)
        gates = F.softmax(self.linear(torch.cat([contexts, title_context, comment_context], dim=-1)), dim=-1)
        return gates


class GetUser(nn.Module):

    def __init__(self, config):
        super(GetUser, self).__init__()
        self.linear = nn.Linear(config.encoder_hidden_size * 2, 10)
        self.use_emb = nn.Parameter(torch.Tensor(10, config.encoder_hidden_size * 2))

    def forward(self, latent_context, is_test=False):
        if not is_test:
            p_user = F.softmax(self.linear(latent_context), dim=-1)  # bsz * 10
            h_user = (self.use_emb.unsqueeze(0) * p_user.unsqueeze(-1)).sum(dim=1)  # bsz * n_hidden
        else:
            ids = torch.LongTensor(latent_context.size(0)).to(latent_context.device).random_(0, 10)
            # h_user = self.use_emb(ids)
            h_user = torch.index_select(self.use_emb, 0, ids)
        return h_user

class var_select_var_user_diverse2seq(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(var_select_var_user_diverse2seq, self).__init__()
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
        self.title_encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.select_gate = SelectGate(config)
        self.comment_encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.hidden_to_mu = nn.Linear(2 * config.encoder_hidden_size, config.n_z)
        self.hidden_to_logvar = nn.Linear(2 * config.encoder_hidden_size, config.n_z)
        self.gama_kld = config.gama_kld

        self.select_post_gate = PostSelectGate(config)
        self.gama_kld_select = config.gama_select
        self.get_user = GetUser(config)

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs'].transpose(0, 1)
        word_loss = models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

        # gate loss
        gate_loss = out_dict['l1_gates']

        # # match loss
        # pos_loss = torch.log(torch.sigmoid((out_dict['title_state'] * out_dict['comment_state']).sum(dim=-1)))
        # neg_losg = torch.log(torch.sigmoid((out_dict['title_state'] * torch.roll(out_dict['comment_state'], 1, dims=0)).sum(dim=-1)))
        # match_loss = - pos_loss + neg_losg

        # kld comment
        kld = out_dict['kld']

        # kld select
        kld_select = out_dict['kld_select']

        # rank and reg loss
        rank_loss = out_dict['rank']
        reg_loss = out_dict['reg']

        loss = word_loss[0] + self.config.gama1 * gate_loss + self.gama_kld * kld + self.gama_kld_select * kld_select \
        + self.config.gama_rank * rank_loss + self.config.gama_reg * reg_loss
        return {
            'loss': loss,
            'word_loss': word_loss[0],
            'acc': word_loss[1],
            'gate_loss': gate_loss,
            'kld_loss': kld,
            'kld_select_loss': kld_select,
            'rank': rank_loss,
            'reg': reg_loss,
        }

    def encode(self, batch, is_test=False):
        src, src_len, src_mask = batch.title, batch.title_len, batch.title_mask
        # content, content_len, content_mask = batch.content, batch.cotent_len, batch.cotent_mask
        content, content_len, content_mask = batch.title_content, batch.title_content_len, batch.title_content_mask

        # input: title, content
        title_contexts, title_state = self.title_encoder(src, src_len)
        title_rep = title_state[0][-1]  # bsz * n_hidden

        # encoder
        contexts, state = self.encoder(content, content_len)

        # select important information of body
        context_gates = self.select_gate(contexts, title_rep)  # output: bsz * n_context * 2
        context_gates = context_gates[:, :, 0]  # bsz * n_context

        if not is_test:
            # comment encoder
            tgt, tgt_len = batch.tgt, batch.tgt_len
            _, comment_state = self.comment_encoder(tgt, tgt_len)  # output: bsz * n_hidden
            comment_rep = comment_state[0][-1]  # bsz * n_hidden

            # selector vae
            org_post_context_gates = self.select_post_gate(contexts, title_rep, comment_rep)
            post_context_gates = gumbel_softmax(torch.log(org_post_context_gates + 1e-10), self.config.tau)
            post_context_gates = post_context_gates[:, :, 0]  # bsz * n_context
            org_post_context_gates = org_post_context_gates[:, :, 0]

            # kl(p1||p2)
            def kldiv(p1, p2):
                kl = p1 * torch.log((p1 + 1e-10) / (p2 + 1e-10)) + (1 - p1) * torch.log((1 - p1 + 1e-10) / (1 - p2 + 1e-10))
                return kl

            kld_select = (kldiv(org_post_context_gates, context_gates) * content_mask.float()).sum() / content_mask.nonzero().size(0)

            # comment vae
            mu = self.hidden_to_mu(comment_rep)  # Get mean of lantent z
            logvar = self.hidden_to_logvar(comment_rep)  # Get log variance of latent z

            z = torch.randn([comment_rep.size(0), self.config.n_z]).to(mu.device)  # Noise sampled from Normal(0,1)
            z = mu + z * torch.exp(0.5 * logvar)  # Reparameterization trick
            kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()  # Compute KL divergence loss
        else:
            comment_rep = None
            z = torch.randn([contexts.size(0), self.config.n_z]).to(contexts.device)
            kld = 0.0
            kld_select = 0.0

            post_context_gates = context_gates

        return contexts, state, post_context_gates, z, kld, title_rep, comment_rep, kld_select

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, post_context_gates, z, kld, title_rep, comment_rep, kld_select = self.encode(batch)

        content_len, content_mask = batch.title_content_len, batch.title_content_mask
        tgt, tgt_len = batch.tgt, batch.tgt_len

        # get user
        h_user = self.get_user(z)
        h_user_neg = torch.roll(h_user, 1, dims=0)
        # user loss
        rank_loss = (1 - torch.sum(h_user*z, dim=-1) + torch.sum(h_user_neg*z, dim=-1)).clamp(min=0).mean()
        reg_loss = torch.mm(self.get_user.use_emb, self.get_user.use_emb.t()) - torch.eye(10, dtype=h_user.dtype, device=h_user.device)
        reg_loss = torch.norm(reg_loss, 2, dim=-1).mean()

        # decoder
        outputs, final_state, attns = self.decoder(tgt[:, :-1], state, contexts, post_context_gates, h_user)
        # return outputs, gates, title_state[0], comment_state[0]

        l1_gates = (post_context_gates * content_mask.float()).sum(dim=-1) / content_len.float()
        return {
            'outputs': outputs,
            'l1_gates': l1_gates.mean(),
            'title_state': title_rep,
            'comment_state': comment_rep,
            'kld': kld,
            'kld_select': kld_select,
            'rank': rank_loss,
            'reg': reg_loss,
        }

    def sample(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, context_gates, z, _, _, _, _ = self.encode(batch, True)
        h_user = self.get_user(z, True)

        bos = torch.ones(contexts.size(0)).long().fill_(self.vocab.word2id('[START]'))
        bos = bos.to(contexts.device)
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts, context_gates, h_user)

        return sample_ids, final_outputs[1]

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1):
        # (1) Run the encoder on the src. Done!!!!
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, enc_state, context_gates, z, _, _, _, _ = self.encode(batch, True)
        h_user = self.get_user(z, True)

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
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

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