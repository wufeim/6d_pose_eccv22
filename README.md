# Robust Category-Level 6D Pose Estimation with Coarse-to-Fine Rendering of Neural Features

PyTorch implementation for robust category-level 6D pose estimation with coarse-to-fine rendering of neural features.

<div align="center">
  <img src="imgs/6d_pose_eccv22.png" width="100%">
</div>

**Robust Category-Level 6D Pose Estimation with Coarse-to-Fine Rendering of Neural Features**
<br />
[Wufei Ma](https://wufeim.github.io),
[Angtian Wang](https://github.com/Angtian),
[Alan Yuille](https://www.cs.jhu.edu/~ayuille/),
[Adam Kortylewski](https://adamkortylewski.com/)
<br />
In European Conference on Computer Vision (ECCV) 2022
<br />
[[Paper]]()

## Installation

Follow ```requirements.txt``` to setup the environment.
- [PyTorch](https://pytorch.org)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

## Preparation

1. Prepare PASCAL3D+ with ```./prepare_data.sh```.

## Demo

6D pose estimation with multi-object reasoning:
```sh
CUDA_VISIBLE_DEVICES=0 python3 tools/demo.py
```
