import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from qpth.qp import QPFunction

from model.models import FewShotModel
from model.utils import one_hot
from model.networks.Eplstm import DNDLSTMCell


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones((in_features, out_features), device=torch.device('cuda')), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), device=torch.device('cuda')), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(adj, torch.matmul(x, self.weight))
        if self.bias is not None:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LinearIter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearIter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        lstm_in = x.unsqueeze(0)
        out = self.mlp(lstm_in)
        return out.squeeze()

    def resetnode(self):
        pass


class DNDLSTMMod(nn.Module):
    def __init__(self, in_dim, out_dim, lstmdict=2000):
        super(DNDLSTMMod, self).__init__()
        self.lst = DNDLSTMCell(in_dim, out_dim, lstmdict)
        self.odim = out_dim
        self.hid = (torch.zeros((1, out_dim), device=torch.device('cuda')), torch.zeros((1, out_dim), device=torch.device('cuda')))
        self.fc = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        lstm_input = x.unsqueeze(0)
        h, c = self.lst(lstm_input, self.hid[0], self.hid[1])
        self.hid = (h, c)
        out = self.fc(h)
        return out.squeeze()

    def resetnode(self):
        self.hid = (torch.zeros((1, self.odim), device=torch.device('cuda')), torch.zeros((1, self.odim), device=torch.device('cuda')))


class GraphFunc(nn.Module):
    def __init__(self, z_dim, way=5, shot=5, query=15, eval_way=5, eval_shot=5, eval_query=15):
        super(GraphFunc, self).__init__()
        self.way, self.shot, self.query = way, shot, query
        self.eway, self.eshot, self.equery = eval_way, eval_shot, eval_query
        self.emb_dim = z_dim
        self.gc1 = GraphConvolution(z_dim, z_dim * 2)
        self.gc2 = GraphConvolution(z_dim * 2, z_dim)
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(self.graph_dim, 1, bias=False)

    def transfunc(self, graph_input, adj):
        graph_out = self.relu(self.gc1(graph_input, adj))
        graph_out = self.gc2(graph_out, adj) + graph_input
        return graph_out

    @staticmethod
    def graph_normalize(mx):
        rowsum = torch.sum(mx, dim=1)
        r_inv_0 = torch.zeros_like(rowsum)
        r_inv = torch.pow(rowsum, -0.5)
        r_mat_inv = torch.where(torch.isinf(r_inv), r_inv_0, r_inv)
        r_mat_inv = torch.diag_embed(r_mat_inv)
        mx_normal = torch.matmul(torch.matmul(r_mat_inv, mx), r_mat_inv)
        return mx_normal

    def forward(self, support, query):
        support, query = support.squeeze(0), query.squeeze(0)
        if self.training:
            graph_dim = self.way + self.query * self.way
            way, shot, querynum = self.way, self.shot, self.query
        else:
            graph_dim = self.eway + self.equery * self.eway
            way, shot, querynum = self.eway, self.eshot, self.equery
        support = torch.mean(support, dim=0)
        adj = torch.zeros((graph_dim, graph_dim), device=torch.device('cuda'))
        # sim_matrix = torch.matmul(F.normalize(support, dim=-1), query.t())
        sim_matrix = torch.ones((way, querynum * way), device=torch.device('cuda'))
        concatpos = way
        adj[:concatpos, concatpos:] = sim_matrix
        adj[concatpos:, :concatpos] = sim_matrix.t()
        adj[:concatpos, :concatpos] = torch.ones((way, way), device=torch.device('cuda'))
        adjm = self.graph_normalize(adj + torch.eye(graph_dim, device=torch.device('cuda')))
        graph_in = torch.cat([support, query], dim=0)
        graph_out = self.transfunc(graph_in, adjm)
        support_out = graph_out[way:]
        support_out = torch.mean(support_out, dim=0).squeeze()
        return support_out


class GCN(FewShotModel):
    def __init__(self, args):
        super(GCN, self).__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim, lstmindim, lstmoutdim = 64, 64 + 64 * 4 + 1, 64 * 4
        elif args.backbone_class == 'Res12':
            hdim, lstmindim, lstmoutdim = 640, 640 + (320 + 640) * 2 + 1, (320 + 640) * 2
        else:
            raise ValueError('No such backbone network.')
        self.emb_dim = hdim
        self.way, self.shot, self.query = args.way, args.shot, args.query
        self.eway, self.eshot, self.equery = args.eval_way, args.eval_shot, args.eval_query
        self.concat_dim = lstmoutdim // 2
        self.lst = DNDLSTMMod(lstmindim, lstmoutdim, args.dictlen)
        self.loss_cls = nn.CrossEntropyLoss()
        self.transfunc = GraphFunc(hdim, args.way, args.shot, args.query, args.eval_way, args.eval_shot, args.eval_query)
        self.T = args.temperature
        self.balance = args.balance
        self.iter = args.rein_iter
        self.discount = args.discount
        self.rewb = args.reward_b
        self.method = args.method

    @staticmethod
    def rein_prob(x, mu, sigma_sq):
        pi = torch.tensor([math.pi], device=torch.device('cuda'))
        a = torch.exp(-1. * torch.pow(x - mu, 2.) / (2. * sigma_sq))
        b = 1. / torch.sqrt(2. * sigma_sq * pi.expand_as(sigma_sq))
        return torch.log(a * b)

    def potential_f(self, state, query):
        logits = self.calculate_loss(self.method, state, query, None, True)
        logits = F.softmax(logits, dim=-1)
        entropy = torch.mean(torch.sum(logits * torch.log(logits + 1e-6), dim=-1))
        return entropy

    def reward_shaping(self, state_new, state_old, query, discount):
        sc = torch.mean(self.potential_f(state_new, query) * discount - self.potential_f(state_old, query))
        return sc

    def calculate_loss_proto(self, support_out, query, labels, return_logits=False):
        support_out, query = support_out.squeeze(0), query.squeeze(0)
        proto_output = F.normalize(torch.mean(support_out, dim=0), dim=-1)
        logits = torch.matmul(query, proto_output.t()) / self.T
        if not return_logits:
            loss = self.loss_cls(logits, labels)
            return loss
        else:
            return logits

    def calculate_loss(self, kernel, support_out, query, support_labels, query_labels, return_logits=False):
        if kernel == 'proto':
            return self.calculate_loss_proto(support_out, query, query_labels, return_logits=return_logits)
        elif kernel == 'subspace':
            return self.calculate_loss_subspace(support_out, query, query_labels, return_logits=return_logits)
        elif kernel == 'svm':
            return self.calculate_loss_svm(support_out, query, support_labels, query_labels, return_logits=return_logits)
        else:
            raise NotImplementedError('No such method.')

    def calculate_loss_subspace(self, support_out, query, labels, lam=0.03, return_logits=False):
        support_out, query = support_out.squeeze(0), query.squeeze(0)
        support_out = support_out.transpose(0, 1)
        dists, hyperplanes = [], []
        if self.training:
            way, shot = self.way, self.shot
        else:
            way, shot = self.eway, self.eshot
        subdim = shot - 1 if shot > 1 else 1
        for ww in range(way):
            the_class = support_out[ww]
            the_class2 = the_class - torch.mean(the_class, dim=0).unsqueeze(0).expand_as(the_class).clone() if shot > 1 else the_class
            the_class2 = the_class2.transpose(0, 1)
            u, _, _ = torch.linalg.svd(the_class2.double())
            u = u.float()
            subspace = u[:, :subdim]
            hyperplanes.append(subspace)
            projection = torch.matmul(subspace, torch.matmul(subspace.transpose(0, 1), query.transpose(0, 1))).transpose(0, 1)
            dist_perclass = torch.sum(torch.pow(query - projection, 2.), dim=-1)
            dists.append(dist_perclass)
        dists = torch.stack(dists, dim=0).transpose(0, 1)
        logits = -1. * dists / self.emb_dim
        if not return_logits:
            discrimin_loss = 0.
            for qq in range(way):
                for ww in range(qq + 1, way):
                    dloss = torch.sum(torch.pow(torch.mm(hyperplanes[qq].t(), hyperplanes[ww]), 2.))
                    discrimin_loss += dloss
            loss = self.loss_cls(logits, labels) + lam * discrimin_loss
            return loss
        else:
            return logits

    def calculate_loss_svm(self, support_out, query, support_labels, query_labels, C_reg=0.1, maxiter=15, return_logits=False):
        support_out, query = support_out.squeeze(0), query.squeeze(0)
        if self.training:
            way, shot, querynum = self.way, self.shot, self.query
        else:
            way, shot, querynum = self.eway, self.eshot, self.equery
        support_out = support_out.view(way * shot, -1)
        # support_out, query = F.normalize(support_out, dim=-1), F.normalize(query, dim=-1)
        kernel_matrix = torch.mm(support_out, support_out.t())
        id_0 = torch.eye(way, device=torch.device('cuda'))
        block_kernel_m = torch.kron(kernel_matrix, id_0)
        block_kernel_m += torch.eye(way * way * shot, device=torch.device('cuda'))
        support_labels_onehot = one_hot(support_labels, way).view(-1)
        G = block_kernel_m
        e = -1. * support_labels_onehot
        C = Variable(torch.eye(way * way * shot, device=torch.device('cuda')))
        h = Variable(C_reg * support_labels_onehot)
        id_1 = torch.eye(way * shot, device=torch.device('cuda'))
        A = Variable(torch.kron(id_1, torch.ones((1, way), device=torch.device('cuda'))))
        b = Variable(torch.zeros(way * shot, device=torch.device('cuda')))
        G, e, C, h, A, b = [ii.double() for ii in [G, e, C, h, A, b]]
        qp_sol = QPFunction(verbose=False, maxIter=maxiter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())
        compat = torch.mm(support_out, query.t())
        compat = compat.unsqueeze(2).expand(way * shot, way * querynum, way).clone()
        qp_sol = qp_sol.view(way * shot, way).float()
        logits = qp_sol.unsqueeze(1).expand(way * shot, way * querynum, way).clone()
        logits *= compat
        logits = torch.sum(logits, dim=0)
        if return_logits:
            return logits
        else:
            loss = self.loss_cls(logits, query_labels)
            return loss

    def policy(self, lstmr_input):
        action = self.lst(lstmr_input)
        action_ru, action_bu = action[self.concat_dim:] + torch.ones((self.concat_dim,), device=torch.device('cuda')), action[:self.concat_dim]
        return action_ru, action_bu

    def _forward(self, instance, support_idx, query_idx, labels):
        emb_dim = self.emb_dim
        support_num, query_num = support_idx.size(1) * support_idx.size(2), query_idx.size(1) * query_idx.size(2)
        if self.training:
            reward_losses, act_prob = [], []
            lossrein, loss_init = 1., 0.
            self.lst.resetnode()
            scalea, shifta = torch.ones((self.concat_dim,), device=torch.device('cuda')), torch.zeros((self.concat_dim,), device=torch.device('cuda'))
            for i in range(self.iter + 1):
                instance_embs = self.enc2(instance, scalea, shifta)
                support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.size() + (emb_dim,)))  # (1, shot, way, d)
                query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(query_idx.size(0), query_num, emb_dim)  # (1, query x way, d)
                if i == 0:
                    loss_init = self.calculate_loss(self.method, support, query, labels[1], labels[0]) + \
                                self.calculate_loss_proto(support, support.view(support_num, -1), labels[1]) * self.balance
                else:
                    reward_losses.append(self.calculate_loss(self.method, support, query, labels[1], labels[0]))
                    act_prob.append(lossrein)
                if i < self.iter:
                    task_state = self.transfunc(support, query)
                    tflags = torch.zeros((1,), device=torch.device('cuda')) if i < self.iter - 1 else torch.ones((1,), device=torch.device('cuda'))
                    lstmr_input = torch.cat([task_state, scalea, shifta, tflags], dim=0)
                    scaleau, shiftau = self.policy(lstmr_input)
                    combau = torch.cat([scaleau, shiftau], dim=0)
                    comba = torch.randn(combau.size(), device=torch.device('cuda')) + combau
                    scalea, shifta = comba[:self.concat_dim], comba[self.concat_dim:]
                    lossrein = -1. * torch.mean(self.rein_prob(comba, combau, torch.ones(combau.size(), device=torch.device('cuda'))))
            if self.iter > 0:
                accm_reward, accum = 0., []
                for r in reversed(reward_losses):
                    accm_reward = accm_reward * self.discount + r
                    accum.insert(0, accm_reward)
                accum, act_prob = torch.stack(accum, dim=0), torch.stack(act_prob, dim=0)
                reward_loss = torch.sum(accum * act_prob) * self.rewb
            else:
                reward_loss = torch.zeros((1,), device=torch.device('cuda'))
            return reward_loss, loss_init
        else:
            self.lst.resetnode()
            scaleau, shiftau = torch.ones((self.concat_dim,), device=torch.device('cuda')), torch.zeros((self.concat_dim,), device=torch.device('cuda'))
            for i in range(self.iter + 1):
                instance_embs = self.enc2(instance, scaleau, shiftau)
                support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.size() + (emb_dim,)))  # (1, shot, way, d)
                query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(query_idx.size(0), query_num, emb_dim)  # (1, query x way, d)
                if i < self.iter:
                    task_state = self.transfunc(support, query)
                    tflags = torch.zeros((1,), device=torch.device('cuda')) if i < self.iter - 1 else torch.ones((1,), device=torch.device('cuda'))
                    lstmr_input = torch.cat([task_state, scaleau, shiftau, tflags], dim=0)
                    scaleau, shiftau = self.policy(lstmr_input)
            logits = self.calculate_loss(self.method, support, query, labels, None, return_logits=True)
            return logits
