import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class tieredImageNet(Dataset):
    def __init__(self, setname: str, args, augment: bool = False, data_path: str = 'data/tiered_imagenet'):
        labels = os.listdir(os.path.join(data_path, setname))
        self.label, self.data = [], []
        lb = 0
        for wnid in labels:
            for _, _, ff in os.walk(os.path.join(data_path, setname, wnid)):
                for fff in ff:
                    self.data.append(os.path.join(data_path, setname, wnid, fff))
                    self.label.append(lb)
                lb += 1
        self.num_class = len(set(self.label))
        if augment and setname == 'train':
            transforms_list = [
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
            ]
        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                                        std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])]
            )
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def __getitem__(self, index: int):
        imgp, label = self.data[index], self.label[index]
        img = self.transform(Image.open(imgp).convert('RGB'))
        return img, label

    def __len__(self):
        return len(self.data)
