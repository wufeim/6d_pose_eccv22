from datetime import datetime
import logging
import os
import random

import cv2
import math
import numpy as np
from PIL import Image
import torch

from .pose import pose_error


def get_config_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, "..", "..", "configs"))
    return root


def setup_logging(save_path):
    os.makedirs(os.path.join(save_path, 'logs'), exist_ok=True)
    dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(save_path, 'logs', f'log_{dt}.txt'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger('').handlers[0].baseFilename


def normalize_features(x, dim=0):
    return x / torch.sum(x ** 2, dim=dim, keepdim=True)[0] ** .5


def load_image(img_path, h, w):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    resize_rate = float(min(h / img.shape[0], w / img.shape[1]))
    dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
    img = cv2.resize(img, dsize=dsize)

    assert img.shape[0] <= h and img.shape[1] <= w

    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
    if img.shape[0] < h:
        pad_v = h - img.shape[0]
        pad_t, pad_b = pad_v//2, pad_v-pad_v//2
    if img.shape[1] <= w:
        pad_h = w - img.shape[1]
        pad_l, pad_r = pad_h//2, pad_h-pad_h//2
    img = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant')

    return img


def get_anno(record, *args, idx=0):
    out = []
    for key_ in args:
        if key_ == 'class':
            out.append(record['objects'][0, 0]['class'][0, idx])
        elif key_ == 'difficult':
            out.append(record['objects'][0, 0]['difficult'][0, idx])
        elif key_ == 'height':
            out.append(record['imgsize'][0, 0][0][1])
        elif key_ == 'width':
            out.append(record['imgsize'][0, 0][0][0])
        elif key_ == 'bbox':
            out.append(record['objects'][0, 0]['bbox'][0, idx][0])
        elif key_ == 'cad_index':
            if len(record['objects'][0, 0]['cad_index'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['cad_index'][0, idx][0, 0])
        elif key_ == 'principal':
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(np.array([None, None]))
            else:
                px = record['objects'][0, 0]['viewpoint'][0, idx]['px'][0, 0][0, 0]
                py = record['objects'][0, 0]['viewpoint'][0, idx]['py'][0, 0][0, 0]
                out.append(np.array([px, py]))
        elif key_ in ['theta', 'azimuth', 'elevation']:
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['viewpoint'][0, idx][key_][0, 0][0, 0] * math.pi / 180)
        else:
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['viewpoint'][0, idx][key_][0, 0][0, 0])

    if len(out) == 1:
        return out[0]

    return tuple(out)


def evaluate(gt_sample, pred, metrics=None):
    if metrics is None:
        metrics = ['pose_error']

    gt_pose = {'azimuth': gt_sample['azimuth'], 'elevation': gt_sample['elevation'], 'theta': gt_sample['theta']}
    if 'distance' in gt_sample:
        gt_pose['distance'] = gt_sample['distance']
    else:
        gt_pose['distance'] = 5.0

    pred_pose = {'azimuth': pred['azimuth'], 'elevation': pred['elevation'], 'theta': pred['theta']}
    if 'distance' in pred:
        pred_pose['distance'] = pred['distance']
    else:
        pred_pose['distance'] = 5.0

    result = {}
    if 'pose_error' in metrics:
        result['pose_error'] = pose_error(gt_pose, pred_pose)

    return result


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
    logging.info(f'Set random seed to {seed}')
