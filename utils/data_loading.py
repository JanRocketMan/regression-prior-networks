"""A code to load & augment Nyu v2 or KITTI dataset.
Adapted from https://github.com/ialhashim/DenseDepth
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps, ImageFile
from zipfile import ZipFile
from io import BytesIO
import random
import glob
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def extract_zip(input_zip):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def random_crop_and_resize(image, target_shape, seed=None):
    w, h = image.size
    if seed is not None:
        random.seed(seed)
    if (h > w * target_shape[1] / target_shape[0]):
        # Randomly crop height dimension
        new_height = int(w * target_shape[1] / target_shape[0])
        if new_height != h:
            yleft = random.randrange(0, h - new_height)
        else:
            yleft = 0
        return image.crop((0, yleft, w, yleft + new_height)).resize(target_shape)
    elif (w > h * target_shape[0] / target_shape[1]):
        # Randomly crop width dimension
        new_width = int(h * target_shape[0] / target_shape[1])
        if new_width != w:
            xleft = random.randrange(0, w - new_width)
        else:
            xleft = 0
        return image.crop((xleft, 0, xleft + new_width, h)).resize(target_shape)
    else:
        print("got here with shapes", image.size, "desired shape", target_shape)


def load_test_data(zip_path):
    data = extract_zip(zip_path + '/nyu_test.zip')
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    return rgb, depth, crop


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if 'depth' in sample.keys():
            depth = sample['depth']
            if not _is_pil_image(depth):
                raise TypeError(
                    'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if 'depth' in sample.keys():
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        if 'depth' in sample.keys():
            return {'image': image, 'depth': depth}
        else:
            return {'image': image}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image = sample['image']
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(
                    type(image))
            )

        if 'depth' in sample.keys():
            depth = sample['depth']
            if not _is_pil_image(depth):
                raise TypeError('img should be PIL Image. Got {}'.format(
                        type(depth))
                )
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(
                image[
                    ...,
                    list(
                        self.indices[random.randint(0, len(self.indices) - 1)]
                    )
                ]
            )
        if 'depth' in sample.keys():
            return {'image': image, 'depth': depth}
        else:
            return {'image': image}


def loadZipToMem(zip_file, sanity_check=False, is_ood=False):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    if is_ood:
        print('Loading out-of-domain zip file...', end='')
    else:
        print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    l_beg = input_zip.filelist[0].filename
    nyu2_train = list(
        (row.split(',') for row in (
            data[l_beg + 'nyu2_train.csv']
        ).decode("utf-8").split('\n') if len(row) > 0)
    )
    nyu2_val = list(
        (row.split(',') for row in (
            data[l_beg + 'nyu2_test.csv']
        ).decode("utf-8").split('\n') if len(row) > 0)
    )
    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    if sanity_check:
        nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    print('Loaded ({0}).'.format(len(nyu2_val)))
    return data, nyu2_train, nyu2_val


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None, is_ood=False, is_val=False, indata='nyu'):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform
        self.is_ood = is_ood
        self.crop_seed = 42 if is_val else None
        self.target_shapes = (640, 480) if indata == 'nyu' else (1280, 384)

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        if self.is_ood:
            image = Image.open(BytesIO(self.data[sample[0]]))
            if self.target_shapes[0] == 640:
                resized_img = random_crop_and_resize(image, (630, 470), self.crop_seed)
                #resized_img = image.resize((630, 470))
                sample = {'image': ImageOps.expand(resized_img, border=5, fill='white')}
            else:
                resized_img = random_crop_and_resize(image, self.target_shapes, self.crop_seed)
                sample = {'image': resized_img}
        else:
            image = Image.open(BytesIO(self.data[sample[0]]))
            depth = Image.open(BytesIO(self.data[sample[1]]))
            sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


import cv2


class DepthDatasetKITTI(Dataset):
    def __init__(self, data, transform=None):
        """
        data: list with lists of two paths to images:
            [[rgb_path, depth_path], ...]
        """
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.data[idx][0])
        if self.data[idx][1][-3:] == 'png':
            depth = cv2.imread(self.data[idx][1], -1)
            #depth = cv2.imread('/' + self.data[idx][1].split('/')[-1], -1)
            depth = Image.fromarray(np.uint16(depth))
        else:
            depth = np.load(self.data[idx][1])
            depth = Image.fromarray(np.uint16(depth*256))
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)


class ToTensor(object):
    def __init__(self, is_val=False, resize=True):
        self.is_val = is_val
        self.resize = resize

    def __call__(self, sample):
        image = sample['image']

        image = self.to_tensor(image)

        if 'depth' in sample.keys():
            depth = sample['depth']
            if self.resize:
                depth = depth.resize((320, 240))

            if self.is_val:
                depth = self.to_tensor(depth).float() / 10
            else:
                depth = self.to_tensor(depth).float() * 1000

            # put in expected range
            depth = torch.clamp(depth, 0, 1000)
            return {'image': image, 'depth': depth}
        else:
            return {'image': image}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class ToTensorKitti(object):
    def __init__(self, resize_depth=True):
        self.resize_depth = resize_depth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image.resize((1280, 384))
        if self.resize_depth:
            depth = depth.resize((640, 192), resample=Image.NEAREST)
        else:
            depth = depth.resize((1280, 384), resample=Image.NEAREST)  # Eval
        image = np.array(image)/255.
        image = torch.from_numpy(image).float()
        image = image.transpose(0, 1).transpose(0, 2)

        depth = np.array(depth)/256.  # Now we have depth in range [0, 80]
        depth = torch.from_numpy(depth).float()
        depth = depth.unsqueeze(0)

        return {'image': image, 'depth': depth}


def getNoTransform(is_val=False, resize=True):
    return transforms.Compose([
        ToTensor(is_val=is_val, resize=resize)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


def getNoTransformKitti(resize_depth=False):
    return transforms.Compose([
        ToTensorKitti(resize_depth=resize_depth)
    ])


def getDefaultTrainTransformKitti():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensorKitti()
    ])


def getTrainingEvalData(path, batch_size, sanity_check=False, is_ood=False, indata='nyu'):
    data, nyu2_train, nyu2_val = loadZipToMem(
        path, sanity_check=sanity_check, is_ood=is_ood
    )

    transformed_training = depthDatasetMemory(
        data, nyu2_train, transform=getDefaultTrainTransform(), is_ood=is_ood, indata=indata
    )
    transformed_val = depthDatasetMemory(
        data, nyu2_val, transform=getNoTransform(is_val=True), is_ood=is_ood, indata=indata,
        is_val=True
    )

    train_loader = DataLoader(transformed_training, batch_size, shuffle=True)
    val_loader = DataLoader(transformed_val, batch_size, shuffle=False)

    return train_loader, val_loader


def getTrainingEvalDataKITTI(
    path_to_kitti, path_to_csv_train, path_to_csv_val,
    batch_size, resize_depth=True
):
    """
    path_to_kitti: str with path
    path_to_csv:
        csv of format
            rgb,depth
            2011_09_28/2011_09_28_drive_0146_sync/image_02/data/0000000032.png,2011_09_28/2011_09_28_drive_0146_sync/image_02/data/0000000032_depth.png
            ...
    """
    print('Loading KITTI dataset')
    if path_to_kitti[-1] != '/':
        path_to_kitti += '/'
    train_ar = path_to_kitti + pd.read_csv(path_to_csv_train).values
    val_ar = path_to_kitti + pd.read_csv(path_to_csv_val).values

    transformed_training = DepthDatasetKITTI(
        train_ar, transform=getDefaultTrainTransformKitti()
    )
    transformed_val = DepthDatasetKITTI(
        val_ar, transform=getNoTransformKitti(resize_depth=resize_depth)
    )

    train_loader = DataLoader(transformed_training, batch_size, shuffle=True)
    val_loader = DataLoader(transformed_val, batch_size, shuffle=False)

    return train_loader, val_loader
