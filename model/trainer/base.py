import abc
import os.path as osp
from shutil import copyfile
import torch

from model.logger import Logger
from model.utils import Averager, Timer


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args, osp.join(args.save_path, 'tblog'))
        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.timer = Timer()
        self.trlog = {'max_acc': 0., 'max_acc_epoch': 0, 'max_acc_interval': 0., 'test_acc': 0., 'test_acc_interval': 0., 'test_loss': 0.}
        self.model, self.optimizer = None, None

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def evaluate_test(self):
        pass

    @abc.abstractmethod
    def final_record(self):
        pass

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch > args.eval_initepoch and self.train_epoch % args.eval_interval == 0:
            vl, va, vap = self.evaluate()
            self.logger.add_scalar('validation/val_loss', vl, self.train_epoch)
            self.logger.add_scalar('validation/val_acc', va, self.train_epoch)
            print('validation:\nepoch: {}\tloss: {:.4f}\tacc: {:.4f} + {:.4f}'.format(epoch, vl, va, vap))
            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('%d_best_acc' % self.train_epoch, True)
            print('best epoch: {}\tbest val acc: {:.4f} + {:.4f}'.format(self.trlog['max_acc_epoch'], self.trlog['max_acc'], self.trlog['max_acc_interval']))

    def try_logging(self, tl1: Averager, tl2: Averager, ta: Averager):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch: {}\tstep: {}/{}\ttotal: {:.4f}\tceloss: {:.4f}\trewardloss: {:.4f}\tlr: {:.8f}\tbackbonelr: {:.8f}'
                  .format(self.train_epoch, self.train_step, self.max_steps, tl1.item() + tl2.item(), tl1.item(), tl2.item(), args.lr * args.lr_mul, args.lr))
            self.logger.add_scalar('train/train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train/train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train/train_acc', ta.item(), self.train_step)

    def save_model(self, name, is_best: bool = False):
        torch.save(self.model.state_dict(), osp.join(self.args.save_path_ckpt, name + '.pth'))
        if is_best:
            copyfile(osp.join(self.args.save_path_ckpt, name + '.pth'), osp.join(self.args.save_path_ckpt, 'max_acc.pth'))

    def __str__(self):
        return '{} ({})'.format(self.__class__.__name__, self.model.__class__.__name__)
