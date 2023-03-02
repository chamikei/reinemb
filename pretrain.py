import argparse
import os
import os.path as osp
from pprint import pprint
from tqdm import tqdm
from shutil import copyfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model.dataloader.samplers import CategoriesSampler
from model.dataloader.mini_imagenet import MiniImageNet
from model.dataloader.cub import CUB
from model.dataloader.tiered_imagenet import tieredImageNet
from model.models.classifier import Classifier
from model.utils import ensure_path, Averager, Timer, count_acc, set_gpu

# pre-train model, compute validation acc after 500 epoches
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImagenet', 'CUB'])
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12'])
    parser.add_argument('--schedule', type=str, default='75,150,300', help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--query', type=int, default=15)
    args = parser.parse_args()
    args.orig_imsize = -1
    set_gpu(args.gpu)
    save_path = osp.join('saves', 'pretrain_meta', '_'.join([args.dataset, args.backbone_class, str(args.lr), str(args.gamma), '-'.join(args.schedule.split(','))]))
    save_path_tb = osp.join(save_path, 'tblog')
    save_path_ckpt = osp.join(save_path, 'ckpts')
    ensure_path(save_path)
    os.makedirs(save_path_ckpt)

    if args.dataset == 'MiniImageNet':
        trainset = MiniImageNet('train', args, True)
        valset = MiniImageNet('val', args)
    elif args.dataset == 'CUB':
        trainset = CUB('train', args, True)
        valset = CUB('val', args)
    elif args.dataset == 'TieredImagenet':
        trainset = tieredImageNet('train', args, True)
        valset = tieredImageNet('val', args)
    else:
        raise ValueError('No such dataset.')
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_sampler = CategoriesSampler(valset.label, 200, valset.num_class, 1 + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, shuffle=False, num_workers=3, pin_memory=True)
    args.way, args.shot, args.few_way = trainset.num_class, 1, valset.num_class
    pprint(vars(args))
    model = Classifier(args).cuda()
    if 'Conv' in args.backbone_class:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    elif 'Res' in args.backbone_class or 'WRN' in args.backbone_class:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    else:
        raise ValueError('No Such Encoder')
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(k) for k in args.schedule.split(',')], args.gamma)


    def save_checkpoint(ckpt_name, is_best=False):
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, osp.join(save_path_ckpt, ckpt_name + '.pth'))
        if is_best:
            copyfile(osp.join(save_path_ckpt, ckpt_name + '.pth'), osp.join(save_path_ckpt, 'best_model.pth'))


    trlog = {'max_acc_dist': 0., 'max_acc_dist_epoch': 0, 'max_acc_sim': 0., 'max_acc_sim_epoch': 0, 'train_loss': [], 'train_acc': []}
    global_count = 0
    timer = Timer()
    writer = SummaryWriter(logdir=save_path_tb)
    tl, ta = Averager(), Averager()
    vl_dist = Averager()
    va_dist = Averager()
    vl_sim = Averager()
    va_sim = Averager()
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        tl.reset()
        ta.reset()
        i = 0
        for batch in train_loader:
            global_count += 1
            data, label = batch[0].cuda(), batch[1].cuda()
            logits = model(data)
            loss = criterion(logits, label)
            acc = count_acc(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('training/loss', loss.item(), global_count)
            writer.add_scalar('training/acc', acc, global_count)
            tl.add(loss.item())
            ta.add(acc)
            if (i - 1) % 100 == 0:
                print('\nepoch: {}\tstep: {}/{}\tloss: {:.6f}\tacc: {:.4f}%'.format(epoch, i, len(train_loader), loss.item(), acc * 100.))
            i += 1
        writer.add_scalar('training/loss_epoch_avg', tl.item(), epoch)
        writer.add_scalar('training/acc_epoch_avg', ta.item(), epoch)
        trlog['train_loss'].append(tl.item())
        trlog['train_acc'].append(ta.item())
        scheduler.step()
        if epoch > 300 and (epoch - 1) % 5 == 0 or epoch == 1:
            model.eval()
            vl_dist.reset()
            va_dist.reset()
            vl_sim.reset()
            va_sim.reset()
            print('[Dist] best epoch {}\tcurrent best val acc {:.6f}'.format(trlog['max_acc_dist_epoch'], trlog['max_acc_dist']))
            print('[Sim] best epoch {}\tcurrent best val acc {:.6f}'.format(trlog['max_acc_sim_epoch'], trlog['max_acc_sim']))
            label = torch.arange(valset.num_class, device=torch.device('cuda')).repeat(args.query)
            with torch.no_grad():
                for batch, _ in tqdm(val_loader, desc='few-shot validating'):
                    data = batch.cuda()
                    data_shot, data_query = data[:valset.num_class], data[valset.num_class:]
                    logits_dist, logits_sim = model.forward_proto(data_shot, data_query)
                    loss_dist = criterion(logits_dist, label)
                    acc_dist = count_acc(logits_dist, label)
                    loss_sim = criterion(logits_sim, label)
                    acc_sim = count_acc(logits_sim, label)
                    vl_dist.add(loss_dist.item())
                    va_dist.add(acc_dist)
                    vl_sim.add(loss_sim.item())
                    va_sim.add(acc_sim)
            writer.add_scalar('val/fewshot_loss_dist', vl_dist.item(), epoch)
            writer.add_scalar('val/fewshot_acc_dist', va_dist.item(), epoch)
            writer.add_scalar('val/fewshot_loss_sim', vl_sim.item(), epoch)
            writer.add_scalar('val/fewshot_acc_sim', va_sim.item(), epoch)
            print('few-shot validation:')
            print('epoch: {}\tloss_dist: {:.4f}\tacc_dist: {:.4f}\tloss_sim: {:.4f}\tacc_sim: {:.4f}'.format(
                epoch, vl_dist.item(), va_dist.item(), vl_sim.item(), va_sim.item()))
            if va_dist.item() > trlog['max_acc_dist']:
                trlog['max_acc_dist'] = va_dist.item()
                trlog['max_acc_dist_epoch'] = epoch
                save_checkpoint(str(epoch) + '_dist_%.2f' % (va_dist.item()), True)
            if va_sim.item() > trlog['max_acc_sim']:
                trlog['max_acc_sim'] = va_sim.item()
                trlog['max_acc_sim_epoch'] = epoch
                save_checkpoint(str(epoch) + '_sim_%.2f' % (va_sim.item()), True)
            print('ETA: {}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
        if epoch == args.max_epoch:
            save_checkpoint(str(epoch) + '_epoch_last')
    writer.close()
