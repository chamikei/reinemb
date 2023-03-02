import torch
import torch.nn as nn

from model.networks.convnet import ConvNet
from model.networks.res12 import Res12


class FewShotModel(nn.Module):
    def __init__(self, args):
        super(FewShotModel, self).__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            self.enc1, self.enc2 = ConvNet()
        elif args.backbone_class == 'Res12':
            self.enc1, self.enc2 = Res12()
        else:
            raise ValueError('No such backbone network')

    def split_instances(self):
        args = self.args
        if self.training:
            return (torch.arange(args.way * args.shot).view(1, args.shot, args.way),
                    torch.arange(args.way * args.shot, args.way * (args.shot + args.query)).view(1, args.query, args.way))
        else:
            return (torch.arange(args.eval_way * args.eval_shot).view(1, args.eval_shot, args.eval_way),
                    torch.arange(args.eval_way * args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query)).view(1, args.eval_query, args.eval_way))

    def forward(self, x, labels):
        x = x.squeeze(0)
        inn = self.enc1(x)
        support_idx, query_idx = self.split_instances()
        loss = self._forward(inn, support_idx, query_idx, labels)
        return loss

    def _forward(self, x, support_idx, query_idx, labels):
        raise NotImplementedError('Suppose to be implemented by subclass')
