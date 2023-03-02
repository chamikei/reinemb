import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CUB(Dataset):
    def __init__(self, setname, args, augment: bool = False, data_path: str = 'data/cub'):
        im_size = args.orig_imsize
        self.data_path = data_path
        txt_path = osp.join(self.data_path, 'split', setname + '.csv')
        self.use_im_cache = (im_size != -1)
        if self.use_im_cache:
            cache_path = osp.join(self.data_path, 'cache_{}_{}_{}.pt'.format('CUB', setname, im_size))
            if not osp.exists(cache_path):
                print('Cache miss ... Preprocessing {} ...'.format(setname))
                resize_func = lambda x: x if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(txt_path)
                self.data = [resize_func(Image.open(path).convert('RGB')) for path in data]
                self.label = label
                print('Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label}, cache_path)
            else:
                print('Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data = cache['data']
                self.label = cache['label']
        else:
            self.data, self.label = self.parse_csv(txt_path)
        self.num_class = len(set(self.label))
        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                                        std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def parse_csv(self, txt_path):
        data, label, wnids = [], [], []
        lb = -1
        lines = [x.strip() for x in open(txt_path, 'rt').readlines()][1:]
        for l in lines:
            cc = l.split(',')
            name, wnid = cc[0], cc[1]
            path = osp.join(self.data_path, 'images', name)
            if wnid not in wnids:
                wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)
        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))
        return image, label
