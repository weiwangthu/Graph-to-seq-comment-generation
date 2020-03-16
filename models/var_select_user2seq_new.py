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


class GetUser(nn.Module):

    def __init__(self, config):
        super(GetUser, self).__init__()
        self.linear1 = nn.Linear(config.n_z, int(config.n_z / 2))
        self.linear2 = nn.Linear(int(config.n_z / 2), config.n_topic_num)
        self.use_emb = nn.Embedding(config.n_topic_num, config.n_z)

        self.content_linear1 = nn.Linear(config.decoder_hidden_size, int(config.decoder_hidden_size / 2))
        self.content_linear2 = nn.Linear(int(config.decoder_hidden_size / 2), config.n_topic_num)

        self.topic_id = -1
        self.config = config

    def content_to_user(self, latent_context):
        latent_context = F.tanh(self.content_linear1(latent_context))
        p_user = F.softmax(self.content_linear2(latent_context), dim=-1)  # bsz * 10

        if self.config.one_user:
            p_user = gumbel_softmax(torch.log(p_user + 1e-10), self.config.tau)

        h_user = (self.use_emb.weight.unsqueeze(0) * p_user.unsqueeze(-1)).sum(dim=1)  # bsz * n_hidden
        selected_user = torch.argmax(p_user, dim=-1)
        return h_user, selected_user, p_user

    def forward(self, latent_context, is_test=False):
        if not is_test:
            latent_context = F.tanh(self.linear1(latent_context))
            p_user = F.softmax(self.linear2(latent_context), dim=-1)  # bsz * 10

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

class var_select_user2seq_new(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(var_select_user2seq_new, self).__init__()
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

        # vae bow and cluster
        # encoder
        self.enc_linear1 = nn.Linear(self.vocab_size, 2 * config.decoder_hidden_size)
        self.enc_linear2 = nn.Linear(2 * config.decoder_hidden_size, config.decoder_hidden_size)

        # decoder
        self.dec_linear1 = nn.Linear(config.n_z, self.vocab_size)

        # latent
        self.hidden_to_mu = nn.Linear(config.decoder_hidden_size, config.n_z)
        self.hidden_to_logvar = nn.Linear(config.decoder_hidden_size, config.n_z)
        self.gama_kld = config.gama_kld
        self.gama_select = config.gama_select

        # select gate
        self.select_gate = SelectGate(config)
        self.select_post_gate = PostSelectGate(config)
        self.gama_kld_select = config.gama_kld_select

        self.get_user = GetUser(config)

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs'].transpose(0, 1)
        word_loss = models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

        # vae bow and cluster
        bow_hidden_outputs = out_dict['bow_outputs']
        mask_bow = out_dict['mask']
        bow_word_loss = - (bow_hidden_outputs * mask_bow).sum(dim=-1).mean()

        # kld comment
        kld = out_dict['kld']

        # rank and reg loss
        reg_loss = out_dict['reg']

        p_user = out_dict['p_user']
        select_entropy = p_user * torch.log(p_user + 1e-20)
        select_entropy = select_entropy.sum(dim=1).mean()

        con_p_user = out_dict['con_p_user']
        con_select_entropy = con_p_user * torch.log(con_p_user + 1e-20)
        con_select_entropy = con_select_entropy.sum(dim=1).mean()

        loss = word_loss[0] + self.config.gama_reg * reg_loss + self.config.gama_bow * bow_word_loss + self.gama_kld * kld + self.gama_select * select_entropy

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
            'bow_word_loss': bow_word_loss,
            'acc': word_loss[1],
            'kld_loss': kld,
            'reg': reg_loss,
            'user_norm': out_dict['user_norm'],
            'selected_user': out_dict['selected_user'],
            'select_entropy': select_entropy,
            'con_sel_user': out_dict['con_sel_user'],
            'con_sel_entropy': con_select_entropy,
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
            tgt_bow = batch.tgt_bow
            enc_hidden = torch.tanh(self.enc_linear1(tgt_bow.float()))
            enc_hidden = torch.tanh(self.enc_linear2(enc_hidden))

            # comment vae
            mu = self.hidden_to_mu(enc_hidden)  # Get mean of lantent z
            logvar = self.hidden_to_logvar(enc_hidden)  # Get log variance of latent z

            z = torch.randn([enc_hidden.size(0), self.config.n_z]).to(mu.device)  # Noise sampled from Normal(0,1)
            z = mu + z * torch.exp(0.5 * logvar)  # Reparameterization trick

            # for loss
            z_prior_mean = z.unsqueeze(1) - self.get_user.use_emb.weight.unsqueeze(0)
            kld = - 0.5 * (logvar.unsqueeze(1) - z_prior_mean ** 2)

            # selector vae
            comment_rep = enc_hidden
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
            z = torch.randn([batch.title_content.size(0), self.config.n_z]).to(batch.title_content.device)
            kld = 0.0

            comment_rep = None
            kld_select = 0.0

            # random
            # context_gates = torch.bernoulli(context_gates)

            # gumbel
            # context_gates = gumbel_softmax(torch.log(org_context_gates + 1e-10), self.config.tau)
            # context_gates = context_gates[:, :, 0]

            # best
            context_gates[context_gates > 0.5] = 1.0
            context_gates[context_gates <= 0.5] = 0.0

            post_context_gates = context_gates

        return contexts, state, z, kld, post_context_gates, comment_rep, kld_select, context_gates

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        bow_mask = (batch.tgt_bow > 0).float()
        contexts, state, z, kld, post_context_gates, comment_rep, kld_select, context_gates = self.encode(batch)
        content_len, content_mask = batch.title_content_len, batch.title_content_mask
        tgt, tgt_len = batch.tgt, batch.tgt_len

        # get user
        _, selected_user, p_user = self.get_user(z)
        content_h_user, content_selected_user, content_p_user = self.get_user.content_to_user(state[0][-1])

        # decoder
        outputs, final_state, attns = self.decoder(tgt[:, :-1], state, contexts, post_context_gates, content_h_user)

        dec_hidden = torch.log(torch.softmax(- self.dec_linear1(z), dim=-1) + 0.0001)

        # kld loss
        # kld = torch.mean((p_user.unsqueeze(-1) * kld).sum(dim=1), 0).sum()
        kld = (p_user.unsqueeze(-1) * kld).sum(dim=1).sum(dim=1).mean()

        # user loss
        identity_matrix = torch.eye(self.get_user.use_emb.weight.size(0), dtype=z.dtype, device=z.device)
        reg_loss = torch.mm(self.get_user.use_emb.weight, self.get_user.use_emb.weight.t()) - identity_matrix
        reg_loss = reg_loss * (1 - identity_matrix)
        reg_loss = torch.norm(reg_loss)

        user_norm = torch.norm(self.get_user.use_emb.weight, 2, dim=1).mean()

        l1_gates = (post_context_gates * content_mask.float()).sum(dim=-1) / content_len.float()
        pri_gates = (context_gates * content_mask.float()).sum(dim=-1) / content_len.float()
        return {
            'outputs': outputs,
            'bow_outputs': dec_hidden,
            'kld': kld,
            'mask': bow_mask,
            'reg': reg_loss,
            'user_norm': user_norm,
            'selected_user': selected_user,
            'p_user': p_user,
            'con_sel_user': content_selected_user,
            'con_p_user': content_p_user,
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