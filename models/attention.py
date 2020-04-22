import torch
import torch.nn as nn


class memory_attention(nn.Module):
    def __init__(self, key_size, memory_size, hidden_size):
        super(memory_attention, self).__init__()
        self.w_q = nn.Linear(key_size, hidden_size)
        self.w_m = nn.Linear(memory_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()

    def forward(self, x, memory):
        # x (batch, hidden); context (batch, seq_len, hidden)
        h_x = torch.unsqueeze(self.w_q(x), 1)
        h_c = self.w_m(memory)
        score = torch.squeeze(self.v(self.tanh(h_x + h_c)), 2)
        # score = score.masked_fill_((1-mask.byte()), 1e-6)
        score = self.softmax(score)
        c = torch.sum(memory * torch.unsqueeze(score, -1), 1)
        return c, score


class global_attention(nn.Module):

    def __init__(self, hidden_size, activation=None, fix_gate=False):
        super(global_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.activation = activation
        self.fix_gate = fix_gate

    def forward(self, x, context, mask=None, context_gates=None):
        gamma_h = self.linear_in(x).unsqueeze(2)  # batch * size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)  # batch * time

        if context_gates is not None:
            if self.fix_gate:
                weights = self.softmax(weights)
                weights = weights * context_gates
                weights = self.softmax(weights)
                # weights_temp = torch.exp(weights)
                # weights_temp2 = weights_temp * context_gates
                # if torch.isnan(weights_temp2).any():
                #     raise Exception('1nan error')
                # weights_sum = weights_temp2.sum(dim=-1, keepdim=True) + 0.00001
                # weights = weights_temp2 / weights_sum
                # if torch.isnan(weights).any():
                #     raise Exception('2nan error')
            else:
                weights = weights * context_gates
                weights = self.softmax(weights)
        else:
            weights = self.softmax(weights)  # batch * time

        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # batch * size
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights


class global_gate_attention(nn.Module):

    def __init__(self, hidden_size, activation=None):
        super(global_gate_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.activation = activation

        self.gate_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, context, mask=None, context_gates=None):
        gamma_h = self.linear_in(x).unsqueeze(2)  # batch * size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)  # batch * time

        if context_gates is not None:
            weights = weights * context_gates

        weights = self.softmax(weights)  # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # batch * size
        c_t = c_t * torch.sigmoid(self.gate_layer(x))
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights

class global_attention_with_topic(nn.Module):

    def __init__(self, hidden_size, topic_size, activation=None):
        super(global_attention_with_topic, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2 * hidden_size + topic_size, hidden_size)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, x, context, mask=None, context_gates=None, topic=None):
        gamma_h = self.linear_in(x).unsqueeze(2)  # batch * size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)  # batch * time

        if context_gates is not None:
            weights = weights * context_gates

        weights = self.softmax(weights)  # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # batch * size
        output = self.tanh(self.linear_out(torch.cat([c_t, x, topic], 1)))
        return output, weights


class masked_attention(nn.Module):
    def __init__(self, hidden_size, activation=None):
        super(masked_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, x, context, mask):
        gamma_h = self.linear_in(x).unsqueeze(2)  # batch * size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)  # batch * time
        weights = self.softmax(weights) * mask.float()  # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # batch * size
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights


class Bah_attention(nn.Module):
    def __init__(self, hidden_size, activation=None):
        super(Bah_attention, self).__init__()
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_m = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()

    def forward(self, x, context, mask):
        # x (batch, hidden); context (batch, seq_len, hidden)
        h_x = torch.unsqueeze(self.w_q(x), 1)
        h_c = self.w_m(context)
        score = torch.squeeze(self.v(self.tanh(h_x + h_c)), 2)
        # score = score.masked_fill_((1-mask.byte()), 1e-6)
        score = self.softmax(score)
        c = torch.sum(context * torch.unsqueeze(score, -1), 1)
        return c, score
