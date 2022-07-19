import os

import BboxTools as bbt
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import get_anno


class PASCAL3DPTest(Dataset):
    def __init__(self, config, category, split, transform=None):
        super().__init__()
        self.config = config
        self.img_path = os.path.join(config.root_path, config.img_path, f'{category}_{split}')
        self.anno_path = os.path.join(config.root_path, config.anno_path, f'{category}_{split}')
        self.list_file = os.path.join(config.root_path, config.list_file, f'{category}_{split}_val.txt')
        self.category = category
        self.split = split
        self.image_h = config.image_h
        self.image_w = config.image_w
        self.transform = transform

        self.file_list = [l.strip() for l in open(self.list_file).readlines()]
        self.file_list = sorted(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        fname = self.file_list[item]
        img = Image.open(os.path.join(self.img_path, fname+'.JPEG'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        record = sio.loadmat(os.path.join(self.anno_path, fname.split('.')[0]+'.mat'))['record']

        resize_rate = float(min(self.image_h / img.shape[0], self.image_w / img.shape[1]))

        bbox = get_anno(record, 'bbox', idx=0)
        box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))
        box_ori = box.copy()
        box_ori = box_ori.set_boundary(img.shape[0:2])
        box *= resize_rate

        dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
        img = cv2.resize(img, dsize=dsize)

        center = (img.shape[0] // 2, img.shape[1] // 2)
        out_shape = [self.image_h, self.image_w]

        box1 = bbt.box_by_shape(out_shape, center)
        if out_shape[0] // 2 - center[0] > 0 or out_shape[1] // 2 - center[1] > 0 or out_shape[0] // 2 + center[0] - \
                img.shape[0] > 0 or out_shape[1] // 2 + center[1] - img.shape[1] > 0:
            if len(img.shape) == 2:
                padding = (
                (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)))
            else:
                padding = (
                (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)),
                (0, 0))
            img = np.pad(img, padding, mode='constant')
            box = box.shift([padding[0][0], padding[1][0]])
            box1 = box1.shift([padding[0][0], padding[1][0]])
        img_box = box1.set_boundary(img.shape[0:2])
        box_in_cropped = img_box.box_in_box(box)

        img_cropped = img_box.apply(img)
        proj_foo = bbt.projection_function_by_boxes(box_ori, box_in_cropped, compose=False)

        principal = get_anno(record, 'principal', idx=0)
        principal[0] = proj_foo[1](principal[0])
        principal[1] = proj_foo[0](principal[1])

        azimuth = get_anno(record, 'azimuth', idx=0)
        elevation = get_anno(record, 'elevation', idx=0)
        theta = get_anno(record, 'theta', idx=0)
        distance_orig = get_anno(record, 'distance', idx=0)
        distance = distance_orig / resize_rate
        fine_cad_idx = get_anno(record, 'cad_index', idx=0)

        (y1, y2), (x1, x2) = box_in_cropped.bbox
        sample = {
            'img': img_cropped,
            'img_name': fname,
            'azimuth': azimuth,
            'elevation': elevation,
            'theta': theta,
            'distance_orig': distance_orig,
            'distance': distance,
            'fine_cad_idx': fine_cad_idx,
            'resize_rate': resize_rate,
            'principal': principal,
            'bbox': [x1, y1, x2, y2]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def debug(self, item):
        sample = self.__getitem__(item)
        print(sample['img_name'])
        img = sample['img']
        p1, p2 = sample['principal']
        img = cv2.circle(img, (int(p1), int(p2)), 2, (0, 255, 0), -1)
        [x1, y1, x2, y2] = sample['bbox']
        img = cv2.line(img, (int(x1), int(y1)), (int(x1), int(y2)), (0, 255, 0), 2)
        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y1)), (0, 255, 0), 2)
        img = cv2.line(img, (int(x2), int(y2)), (int(x1), int(y2)), (0, 255, 0), 2)
        img = cv2.line(img, (int(x2), int(y2)), (int(x2), int(y1)), (0, 255, 0), 2)
        Image.fromarray(img).save(f'debug.png')
