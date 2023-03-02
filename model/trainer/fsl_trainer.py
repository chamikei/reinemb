import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from model.trainer.base import Trainer
from model.trainer.helpers import get_dataloader, prepare_model, prepare_optimizer
from model.utils import Averager, count_acc, compute_confidence_interval


class FSLTrainer(Trainer):
    def __init__(self, args):
        super(FSLTrainer, self).__init__(args)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.celoss = nn.CrossEntropyLoss().cuda()

    def prepare_label(self):
        args = self.args
        label = torch.arange(args.way, dtype=torch.long, device=torch.device('cuda')).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.long, device=torch.device('cuda')).repeat(args.shot)
        return label, label_aux

    def train(self):
        args = self.args
        label, label_aux = self.prepare_label()
        tl1, tl2, ta = Averager(), Averager(), Averager()
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if args.fix_BN:
                self.model.enc1.eval()
                self.model.enc2.eval()
            tl1.reset()
            tl2.reset()
            ta.reset()
            for batch, _ in self.train_loader:
                self.train_step += 1
                data = batch.cuda()
                reward, lossinit = self.model(data, (label, label_aux))
                tloss = (reward + lossinit) * args.total_scale
                tl1.add(lossinit.item())
                tl2.add(reward.item())
                ta.add(0)
                self.optimizer.zero_grad()
                tloss.backward(retain_graph=True)
                if args.clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()
                self.try_logging(tl1, tl2, ta)
            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print('ETA: {}/{}'.format(self.timer.measure(), self.timer.measure(self.train_epoch / args.max_epoch)))
        self.save_model('%d_epoch_last' % args.max_epoch)

    def evaluate(self):
        args = self.args
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))
        label = torch.arange(args.eval_way, dtype=torch.long, device=torch.device('cuda')).repeat(args.eval_query)
        label_aux = torch.arange(args.eval_way, dtype=torch.long, device=torch.device('cuda')).repeat(args.eval_shot)
        with torch.no_grad():
            countt = 0
            for batch, _ in tqdm(self.val_loader, desc='Validating'):
                data = batch.cuda()
                logits = self.model(data, label_aux)
                loss = self.celoss(logits, label)
                acc = count_acc(logits, label)
                record[countt, 0] = loss.item()
                record[countt, 1] = acc
                countt += 1
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        return vl, va, vap

    def evaluate_test(self):
        args = self.args
        self.model.load_state_dict(torch.load(osp.join(args.save_path_ckpt, 'max_acc.pth')))
        self.model.eval()
        record = np.zeros((10000, 2))
        label = torch.arange(args.eval_way, dtype=torch.long, device=torch.device('cuda')).repeat(args.eval_query)
        label_aux = torch.arange(args.eval_way, dtype=torch.long, device=torch.device('cuda')).repeat(args.eval_shot)
        print('test:\nbest epoch: {}\tbest val acc: {:.4f} + {:.4f}'.format(self.trlog['max_acc_epoch'], self.trlog['max_acc'], self.trlog['max_acc_interval']))
        with torch.no_grad():
            countt = 0
            for batch, _ in tqdm(self.test_loader, desc='Testing'):
                data = batch.cuda()
                logits = self.model(data, label_aux)
                loss = self.celoss(logits, label)
                acc = count_acc(logits, label)
                record[countt, 0] = loss.item()
                record[countt, 1] = acc
                countt += 1
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl
        print('test acc: {:.4f} + {:.4f}\ttest loss: {:.4f}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'], self.trlog['test_loss']))

    def final_record(self):
        args = self.args
        with open(osp.join(args.save_path, '{:.4f}_{:.4f}.log'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'wt') as f:
            f.write('best epoch: {}\tbest val acc: {:.4f} + {:.4f}\n'.format(self.trlog['max_acc_epoch'], self.trlog['max_acc'], self.trlog['max_acc_interval']))
            f.write('test acc: {:.4f} + {:.4f}\ttest loss: {:.4f}\n'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'], self.trlog['test_loss']))
