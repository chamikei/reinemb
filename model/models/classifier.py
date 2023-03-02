import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import euclidean_metric
from model.networks.convnet import ConvNet
from model.networks.res12 import Res12


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.way, self.shot, self.fewway = args.way, args.shot, args.few_way
        if args.backbone_class == 'ConvNet':
            hdim = 64
            self.enc1, self.enc2 = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            self.enc1, self.enc2 = Res12()
        else:
            raise ValueError('No such backbone network')
        self.fc = nn.Linear(hdim, self.way)
        if args.backbone_class == 'Res12':
            self.r = torch.ones((960,), device=torch.device('cuda'))
            self.b = torch.zeros((960,), device=torch.device('cuda'))
        elif args.backbone_class == 'ConvNet':
            self.r = torch.ones((hdim * 2,), device=torch.device('cuda'))
            self.b = torch.zeros((hdim * 2,), device=torch.device('cuda'))

    def forward(self, data):
        out = self.enc2(self.enc1(data), self.r, self.b)
        return self.fc(out)

    def forward_proto(self, data_shot, data_query):
        proto = self.enc2(self.enc1(data_shot), self.r, self.b)
        proto = torch.mean(proto.reshape(self.shot, self.fewway, -1), dim=0)
        proto = F.normalize(proto, dim=-1)
        query = self.enc2(self.enc1(data_query), self.r, self.b)
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.matmul(query, proto.t())
        return logits_dist, logits_sim
