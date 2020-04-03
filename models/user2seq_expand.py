import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from Data import *
from util.misc_utils import move_to_cuda

import numpy as np


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

class user2seq_expand(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(user2seq_expand, self).__init__()
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

    def compute_loss(self, out_dict, targets):
        hidden_outputs = out_dict['outputs'].transpose(0, 1)
        word_loss = models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

        # rank and reg loss
        rank_loss = out_dict['rank']

        loss = word_loss[0] + self.config.gama_rank * rank_loss
        return {
            'loss': loss,
            'word_loss': word_loss[0],
            'acc': word_loss[1],
            'selected_user': out_dict['selected_user'],
            'rank': rank_loss,
        }

    def encode(self, batch, is_test=False):
        # content, content_len, content_mask = batch.content, batch.cotent_len, batch.cotent_mask
        content, content_len, content_mask = batch.title_content, batch.title_content_len, batch.title_content_mask

        # encoder
        contexts, state = self.encoder(content, content_len)

        if not is_test:
            # comment encoder
            tgt, tgt_len = batch.tgt, batch.tgt_len
            _, comment_state = self.comment_encoder(tgt, tgt_len)  # output: bsz * n_hidden
            comment_rep = comment_state[0][-1]  # bsz * n_hidden

        else:
            comment_rep = None

        return contexts, state, comment_rep

    def topic_attention(self, comment, topics):
        comment = comment.unsqueeze(2)  # bsz * n_hidden * 1
        weights = torch.bmm(topics, comment).squeeze(2)  # bsz * n_topic
        weights = F.softmax(weights, dim=-1)
        return weights

    def forward(self, batch, use_cuda):
        if use_cuda:
            batch = move_to_cuda(batch)
        contexts, state, comment_rep = self.encode(batch)

        tgt, tgt_len = batch.tgt, batch.tgt_len

        # map to multi topic
        topics = self.map_to_latent(state[0][-1])  # output: bsz * n_topic * n_hidden
        topic_gates = self.topic_attention(comment_rep, topics)  # output: bsz * n_topic
        topic_gates = gumbel_softmax(torch.log(topic_gates), self.config.tau)  # bsz * n_topic
        selected_user = torch.argmax(topic_gates, dim=-1)

        select_topic = (topics * topic_gates.unsqueeze(-1)).sum(dim=1)  # bsz * n_hidden

        # decoder
        outputs, final_state, attns = self.decoder(tgt[:, :-1], state, contexts, None, select_topic)

        # match loss
        news_rep = state[0][-1]
        news_rep_neg = torch.roll(news_rep, 1, dims=0)

        # user loss
        rank_loss = (1 - torch.sum(comment_rep * news_rep, dim=-1) + torch.sum(comment_rep * news_rep_neg, dim=-1)).clamp(min=0).mean()

        return {
            'outputs': outputs,
            'selected_user': selected_user,
            'rank': rank_loss,
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

        # map to multi topic
        topics = self.map_to_latent(enc_state[0][-1])
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