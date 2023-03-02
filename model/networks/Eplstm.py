"""
A DND-based LSTM based on ...
Ritter, et al. (2018).
Been There, Done That: Meta-Learning with Episodic Recall.
Proceedings of the International Conference on Machine Learning (ICML).
"""
import torch
import torch.nn as nn
from model.networks.DND import DND

N_GATES = 4


class DNDLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dict_len, kernel='cosine', bias=True):
        super(DNDLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.i2h = nn.Linear(input_dim, (N_GATES + 1) * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, (N_GATES + 1) * hidden_dim, bias=bias)
        self.dnd = DND(dict_len, hidden_dim, kernel)
        self.reset_parameter()

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)

    def forward(self, x_t, h, c):
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x_t = x_t.view(x_t.size(0), -1)
        Wx = self.i2h(x_t)
        Wh = self.h2h(h)
        preact = Wx + Wh
        gates = torch.sigmoid(preact[:, : N_GATES * self.hidden_dim])
        f_t = gates[:, :self.hidden_dim]
        i_t = gates[:, self.hidden_dim:2 * self.hidden_dim]
        o_t = gates[:, 2 * self.hidden_dim:3 * self.hidden_dim]
        r_t = gates[:, -self.hidden_dim:]
        c_t_new = torch.tanh(preact[:, N_GATES * self.hidden_dim:])
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_t_new)
        m_t = torch.tanh(self.dnd.get_memory(x_t))
        c_t = c_t + torch.mul(r_t, m_t)
        h_t = torch.mul(o_t, torch.tanh(c_t))
        self.dnd.save_memory(x_t, c_t)
        h_t = h_t.view(h_t.size(0), -1)
        c_t = c_t.view(c_t.size(0), -1)
        output = (h_t, c_t)
        return output

    def turn_off_encoding(self):
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()

    def get_all_mems(self):
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V
