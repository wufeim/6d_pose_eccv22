import BboxTools as bbt
import cv2
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class PASCAL3DPSingleTrain(Dataset):
    def __init__(self, img_path, anno_path, list_path, subtypes, transforms=None, weighted=False, enable_cache=True):
        super().__init__()
        self.img_path = img_path
        self.anno_path = anno_path
        self.list_path = list_path
        self.transforms = transforms
        self.subtypes = subtypes
        self.weighted = weighted
        self.enable_cache = enable_cache
        self.cache_img = dict()
        self.cache_anno = dict()

        self.file_list = sum([
            [l.strip() for l in open(os.path.join(list_path, subtype_ + '.txt')).readlines()]
            for subtype_ in self.subtypes],
            []
        )
        self.file_list = sorted(self.file_list)
        # self.file_list = [x for x in self.file_list if 'resize' not in x]
        # np.random.seed(10)
        # np.random.shuffle(self.file_list)
        # self.file_list = self.file_list[:500]
        # for i in range(5):
        #     print(self.file_list[i])

    def _get_filter_pose_func(self, pose_list):
        if type(pose_list) == str:
            pose_list = [True if c == 'T' else False for c in pose_list]
        return lambda x, pose_list_=pose_list, bin_size = 2 * np.pi / len(pose_list): \
            np.any([np.sum((bin_size * i < x) * (x < bin_size * (i + 1))) for i in range(len(pose_list_)) if pose_list_[i]])

    def filter_pose(self, pose_list):
        new_file_list = []
        f = self._get_filter_pose_func(pose_list)
        for name in self.file_list:
            anno = dict(np.load(os.path.join(self.anno_path, name.split('.')[0] + '.npz'), allow_pickle=True))
            if f(anno['azimuth']):
                new_file_list.append(name)
        print(f'found {len(new_file_list)} out of {len(self.file_list)} samples with target pose {pose_list}')
        self.file_list = new_file_list

    def __getitem__(self, item):
        name_img = self.file_list[item]

        if name_img in self.cache_anno:
            img = self.cache_img[name_img]
            anno = self.cache_anno[name_img]
        else:
            img = Image.open(os.path.join(self.img_path, name_img))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            anno = dict(np.load(os.path.join(self.anno_path, name_img.split('.')[0] + '.npz'), allow_pickle=True))
            if self.enable_cache:
                self.cache_img[name_img] = img
                self.cache_anno[name_img] = anno

        box_obj = bbt.from_numpy(anno['box_obj'])
        obj_mask = np.zeros(box_obj.boundary, dtype=np.float32)
        box_obj.assign(obj_mask, 1)

        kp = anno['cropped_kp_list']
        iskpvisible = anno['visible'] == 1

        if self.weighted:
            iskpvisible = iskpvisible * anno['kp_weights']
        iskpvisible = np.logical_and(iskpvisible, np.any(kp >= np.zeros_like(kp), axis=1))
        iskpvisible = np.logical_and(iskpvisible, np.any(kp < np.array([img.size[::-1]]), axis=1))

        kp = np.max([np.zeros_like(kp), kp], axis=0)
        kp = np.min([np.ones_like(kp) * (np.array([img.size[::-1]]) - 1), kp], axis=0)

        pose_ = np.array([5, anno['elevation'], anno['azimuth'], anno['theta']], dtype=np.float32)
        dist = anno['distance']

        sample = {'img': img, 'kp': kp, 'iskpvisible': iskpvisible, 'sample_name': name_img.split('.')[0], 'obj_mask': obj_mask,
                  'box_obj': box_obj.bbox, 'box_obj_shape': box_obj.shape, 'pose': pose_, 'distance': dist, 'raw_img': img, 'azimuth': anno['azimuth'],
                  'elevation': anno['elevation'], 'theta': anno['theta'], 'principal': anno['principal']}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.file_list)
