import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.spatial import distance_matrix
import torch
from torchvision import transforms

from ..utils import load_off


class ToTensor(object):
    def __init__(self):
        self.trans = transforms.ToTensor()

    def __call__(self, sample):
        sample['img'] = self.trans(sample['img'])
        if 'iskpvisible' in sample and not type(sample['iskpvisible']) == torch.Tensor:
            sample['iskpvisible'] = torch.Tensor(sample['iskpvisible'])
        if 'kp' in sample and not type(sample['kp']) == torch.Tensor:
            sample['kp'] = torch.Tensor(sample['kp'])
        if 'distance' in sample and not type(sample['distance']) == torch.Tensor:
            sample['distance'] = torch.Tensor(sample['distance'])
        if 'raw_img' in sample and not type(sample['raw_img']) == torch.Tensor:
            sample['raw_img'] = self.trans(sample['raw_img'])
        if 'obj_mask' in sample and not type(sample['obj_mask']) == torch.Tensor:
            sample['obj_mask'] = torch.Tensor(sample['obj_mask'])
        return sample


class Normalize(object):
    def __init__(self):
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        sample['img'] = self.trans(sample['img'])
        return sample


def hflip(sample, mapping):
    sample['img'] = transforms.functional.hflip(sample['img'])
    sample['obj_mask'] = np.fliplr(sample['obj_mask']).copy()
    if 'raw_img' in sample:
        sample['raw_img'] = transforms.functional.hflip(sample['raw_img'])
    sample['kp'][:, 1] = sample['img'].size[0] - sample['kp'][:, 1] - 1
    sample['kp'] = sample['kp'][mapping]
    sample['iskpvisible'] = sample['iskpvisible'][mapping]
    return sample


class RandomHorizontalFlip(object):
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.prepare_mapping()
        self.trans = transforms.RandomApply([lambda x, y=self.mapping: hflip(x, y)], p=0.5)

    def prepare_mapping(self):
        xvert, _ = load_off(self.mesh_path)
        xvert_prime = xvert.copy()
        xvert_prime[:, 0] = -xvert[:, 0]
        dist_mat = distance_matrix(xvert, xvert_prime)
        self.mapping = np.argmin(dist_mat, axis=1)

    def __call__(self, sample):
        sample = self.trans(sample)

        return sample


class RandomTranslate(object):
    def __init__(self, max_translate_width, max_translate_height):
        self.max_translate_width = max_translate_width
        self.max_translate_height = max_translate_height

    def __call__(self, sample):
        fill = 0
        if isinstance(sample['img'], torch.Tensor):
            fill = [float(fill)] * transforms.functional._get_image_num_channels(sample['img'])
        img_size = transforms.functional.get_image_size(sample['img'])
        max_dx = float(self.max_translate_width * img_size[0])
        max_dy = float(self.max_translate_height * img_size[1])
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        sample['img'] = transforms.functional.affine(sample['img'], angle=0, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0))
        sample['obj_mask'] = shift(sample['obj_mask'], (ty, tx), cval=0)
        if 'raw_img' in sample:
            sample['raw_img'] = transforms.functional.affine(sample['raw_img'], angle=0, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0))
        sample['kp'] += np.array([ty, tx])
        sample['iskpvisible'] = np.logical_and(sample['iskpvisible'], np.all(sample['kp'] >= np.zeros_like(sample['kp']), axis=1))
        sample['iskpvisible'] = np.logical_and(sample['iskpvisible'], np.all(sample['kp'] < np.array([sample['img'].size[::-1]]), axis=1))
        sample['kp'] = np.max([np.zeros_like(sample['kp']), sample['kp']], axis=0)
        sample['kp'] = np.min([np.ones_like(sample['kp']) * (np.array([sample['img'].size[::-1]]) - 1), sample['kp']], axis=0)

        return sample


class ColorJitter(object):
    def __init__(self):
        self.trans = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0)

    def __call__(self, sample):
        sample['img'] = self.trans(sample['img'])
        return sample
