import os
import argparse
import torch
from pprint import pprint

from model.trainer.fsl_trainer import FSLTrainer
from model.utils import set_gpu, ensure_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--backbone_class', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
    parser.add_argument('--method', type=str, default='proto', choices=['proto', 'subspace', 'svm'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImageNet', 'CUB'])
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--eval_shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--rein_iter', type=int, default=5)
    parser.add_argument('--dictlen', type=int, default=2000)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--balance', type=float, default=0.05)
    parser.add_argument('--reward_b', type=float, default=0.02)
    parser.add_argument('--clip', type=float, default=-1)
    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1, help='-1 for nocache, -2 for no resize, only for miniimagenet and cub')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--total_scale', type=float, default=0.9)
    parser.add_argument('--step_size', type=str, default='30')
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--fix_BN', action='store_true', default=False, help='means we do not update the running mean/var in BN, not to freeze BN')
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--init_weights', type=str, default=None)
    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--eval_initepoch', type=int, default=40)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    save_dir = 'saves/episodic-train-meta'
    save_path1 = '_'.join([args.dataset, args.backbone_class, '{:d}w{:d}s{:d}q'.format(args.way, args.shot, args.query)])
    save_path2 = '_'.join([args.method, '-'.join(args.step_size.split(',')), str(args.gamma), 'lr_{:.2g}_mul_{:.2g}'.format(args.lr, args.lr_mul), str(args.lr_scheduler),
                           'T_{:.2f}'.format(args.temperature), 'D_{:d}'.format(args.dictlen), 'disc_{:.2f}'.format(args.discount), 'iter_{:d}'.format(args.rein_iter),
                           'bsz_{:d}'.format(args.way * (args.shot + args.query))])
    if args.init_weights is not None:
        save_path1 += '_preweight'
    if args.fix_BN:
        save_path2 += '_FixBN'
    if not args.augment:
        save_path2 += '_NoAug'
    args.save_path = os.path.join(save_dir, save_path1, save_path2)
    args.save_path_ckpt = os.path.join(args.save_path, 'ckpts')
    ensure_path(args.save_path)
    os.makedirs(args.save_path_ckpt)
    pprint(vars(args))
    set_gpu(args.gpu)
    torch.cuda.manual_seed(args.seed)
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
