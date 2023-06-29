# -*- coding: utf-8 -*-
import os
import sys
from torchvision import transforms
from torchvision import transforms
from PIL import Image, ImageOps
import torch.utils.data as util_data

def get_image_list(data_dir, classes_list, unseen_classes=[]):
    seen_classes = []
    for i in classes_list:
        if i not in unseen_classes:
            seen_classes.append(i)
    image_list=[]
    if 'Office-31' in data_dir:
        for class_name in seen_classes:
            class_dim = data_dir+'/images/'+class_name
            class_image = os.listdir(class_dim)
            class_file = [class_dim+'/'+i for i in class_image]
            image_list+=class_file
    elif 'AwA' in data_dir:
        for class_name in seen_classes:
            class_dim = data_dir+'/JPEGImages/'+class_name
            class_image = os.listdir(class_dim)
            class_file = [class_dim+'/'+i for i in class_image if 'jpg' in i]
            image_list+=class_file
    elif 'webvision' in data_dir:
        for class_name in seen_classes:
            class_dim = data_dir+'/image/'+class_name
            class_image = os.listdir(class_dim)
            class_file = [class_dim+'/'+i for i in class_image]
            image_list+=class_file
    else:
        for class_name in seen_classes:
            class_dim = data_dir+'/'+class_name
            class_image = os.listdir(class_dim)
            class_file = [class_dim+'/'+i for i in class_image]
            image_list+=class_file
    return image_list

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
    
class PlaceCrop(object):
    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y
    
    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


def make_sample_dic():
    all_sample_dic = {}
    with open("/data00/home/jyzhang/data0/UODTN-master/data/new_AwA2.txt",'r') as f:
        for i in f:
            pos, id = i.split(' ')
            id = int(id)
            all_sample_dic[pos] = id
    with open("/data00/home/jyzhang/data0/UODTN-master/data/WEB_3D3_2.txt",'r') as f:
        for i in f:
            pos, id = i.split(' ')
            id = int(id)
            all_sample_dic[pos] = id
    return all_sample_dic

class CDZS_Dataset(util_data.Dataset):

    def __init__(
        self,
        images_file_path,
        classes_list,
        classes_dic,
        unseen_classes=[],
        split='train',
        args=None,
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.images_file_path = images_file_path
        image_list = get_image_list(images_file_path, classes_list, unseen_classes)
        self.image_list = image_list
        self.split = split
        self.class_dic = classes_dic
        self.args = args
        #self.sample_dic = make_sample_dic()
        if unseen_classes is not None:
            k=0
            self.seen_class_dic = {}
            for i in classes_list:
                if i not in unseen_classes:
                    self.seen_class_dic[i] = k
                    k += 1

        print(f"Number of images in {split}: {len(self.image_list):d}")


    def __getitem__(self, index):
        _img = Image.open(self.image_list[index]).convert("RGB")
        if self.split == "train":
            _img = self.transform_tr(_img)
        elif self.split == "train_target":
            _img = self.transform_tr(_img)
        elif self.split == "val":
            _img = self.transform_val(_img)
        sample={'image': _img,}
        sample["image_name"] = str(self.image_list[index])
        if self.split == "val":
            sample['label'] = self.class_dic[sample["image_name"].split('/')[-2]]
        elif self.split == "train":
            sample['label'] = self.class_dic[sample["image_name"].split('/')[-2]]

        return sample

    def transform_tr(self, sample):
        transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transformer(sample)

    def transform_val(self, sample):
        resize_size = 256
        crop_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([transforms.Resize((resize_size,resize_size)),
                                          PlaceCrop(crop_size, start_center, start_center),
                                          transforms.ToTensor(),
                                          normalize,
                                          ])
        return transformer(sample)
    def __len__(self):
        return len(self.image_list)

    
def load_images(images_file_path, classes_list, name2iid, args=None, batch_size=64, split='train', unseen_classes=[]):
    #image_list = get_image_list(images_file_path, classes_list, unseen_classes)

    if sys.platform == 'win32':
        persistent_workers = True
    else:
        persistent_workers = False

    #persistent_workers = False
    cdzs_dataset = CDZS_Dataset(images_file_path, classes_list, name2iid, unseen_classes=unseen_classes, split=split, args=args)
    if split == 'train':
        images_loader = util_data.DataLoader(cdzs_dataset, batch_size=256, shuffle=True, num_workers=6, prefetch_factor=2, pin_memory=True, drop_last=True, persistent_workers=persistent_workers)
    elif split == 'train_target':
        images_loader = util_data.DataLoader(cdzs_dataset, batch_size=256, shuffle=True, num_workers=6, prefetch_factor=2, pin_memory=True, drop_last=True, persistent_workers=persistent_workers)
    elif split == 'val':
        images_loader = util_data.DataLoader(cdzs_dataset, batch_size=512, shuffle=False, num_workers=6, prefetch_factor=2, pin_memory=True, drop_last=False, persistent_workers=persistent_workers)
    return images_loader