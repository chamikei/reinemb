import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.dataloader.samplers import CategoriesSampler
from model.models.graphnet import GCN
from model.dataloader.mini_imagenet import MiniImageNet
from model.dataloader.cub import CUB
from model.dataloader.tiered_imagenet import tieredImageNet


def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        trainset = MiniImageNet('train', args, args.augment)
        valset = MiniImageNet('val', args)
        testset = MiniImageNet('test', args)
    elif args.dataset == 'CUB':
        trainset = CUB('train', args, args.augment)
        valset = CUB('val', args)
        testset = CUB('test', args)
    elif args.dataset == 'TieredImageNet':
        trainset = tieredImageNet('train', args, args.augment)
        valset = tieredImageNet('val', args)
        testset = tieredImageNet('test', args)
    else:
        raise ValueError('Non-supported Dataset.')
    train_sampler = CategoriesSampler(trainset.label, args.episodes_per_epoch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, num_workers=args.num_workers, batch_sampler=train_sampler, pin_memory=True)
    val_sampler = CategoriesSampler(valset.label, args.num_eval_episodes, args.eval_way, args.eval_shot + args.eval_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    test_sampler = CategoriesSampler(testset.label, 10000, args.eval_way, args.eval_shot + args.eval_query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def prepare_model(args):
    model = GCN(args).cuda()
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights)['state_dict']
        del pretrained_dict['fc.weight'], pretrained_dict['fc.bias']
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def prepare_optimizer(model, args):
    bott_para = list(model.enc1.parameters()) + list(model.enc2.parameters())
    top_para = list(model.lst.parameters()) + list(model.transfunc.parameters())
    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam([{'params': bott_para, 'lr': args.lr}, {'params': top_para, 'lr': args.lr * args.lr_mul}], lr=args.lr)
    else:
        optimizer = optim.SGD([{'params': bott_para, 'lr': args.lr}, {'params': top_para, 'lr': args.lr * args.lr_mul}], momentum=args.mom, nesterov=True,
                              weight_decay=args.weight_decay, lr=args.lr)
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.step_size), gamma=args.gamma)
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(k) for k in args.step_size.split(',')], gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch, eta_min=0)
    else:
        raise ValueError('No Such Scheduler')
    return optimizer, lr_scheduler
