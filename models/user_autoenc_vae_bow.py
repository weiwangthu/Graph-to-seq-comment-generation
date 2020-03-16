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
        self.linear1 = nn.Linear(config.n_z, int(config.n_z / 2))
        self.linear2 = nn.Linear(int(config.n_z / 2), config.n_topic_num)
        self.use_emb = nn.Embedding(config.n_topic_num, config.n_z)
        self.topic_id = -1
        self.config = config

    def forward(self, latent_context, is_test=False):
        if not is_test:
            latent_context = F.tanh(self.linear1(latent_context))
            p_user = F.softmax(self.linear2(latent_context), dim=-1)  # bsz * 10

            if self.config.one_user:
                p_user = gumbel_softmax(torch.log(p_user + 1e-10), self.config.tau)

            selected_user = torch.argmax(p_user, dim=-1)
        else:
            if self.topic_id == -1:
                ids = torch.LongTensor(latent_context.size(0)).to(latent_context.device).random_(0, self.use_emb.weight.size(0))
            else:
                ids = torch.LongTensor(latent_context.size(0)).to(latent_context.device).fill_(self.topic_id)
            selected_user = ids
            p_user = None
        return selected_user, p_user

class user_autoenc_vae_bow(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(user_autoenc_vae_bow, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        self.config = config
        self.use_content = use_content
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax(-1)

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

        self.get_user = GetUser(config)

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs']
        mask_bow = out_dict['mask']
        word_loss = - (hidden_outputs * mask_bow).sum(dim=-1).mean()

        # kld comment
        kld = out_dict['kld']

        # rank and reg loss
        reg_loss = out_dict['reg']

        p_user = out_dict['p_user']
        select_entropy = p_user * torch.log(p_user + 1e-20)
        select_entropy = select_entropy.sum(dim=1).mean()

        loss = word_loss + self.config.gama_reg * reg_loss + self.gama_kld * kld + self.gama_select * select_entropy
        return {
            'loss': loss,
            'word_loss': word_loss,
            'acc': torch.LongTensor([0]),
            'kld_loss': kld,
            'reg': reg_loss,
            'user_norm': out_dict['user_norm'],
            'selected_user': out_dict['selected_user'],
            'select_entropy': select_entropy,

        }

    def encode(self, batch, is_test=False):
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
        else:
            z = torch.randn([batch.title_content.size(0), self.config.n_z]).to(batch.title_content.device)
            kld = 0.0

        return z, kld

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        bow_mask = (batch.tgt_bow > 0).float()
        z, kld = self.encode(batch)

        # get user
        selected_user, p_user = self.get_user(z)

        # decoder
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
        return {
            'outputs': dec_hidden,
            'kld': kld,
            'mask': bow_mask,
            'reg': reg_loss,
            'user_norm': user_norm,
            'selected_user': selected_user,
            'p_user': p_user,

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