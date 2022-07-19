import _init_paths

import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from PIL import Image
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, camera_position_from_spherical_angles
import torch
from torchvision import transforms

from src.models import NetE2E, NearestMemoryManager, MeshInterpolateModule
from src.optim import pre_compute_kp_coords
from src.utils import load_default_config, load_off, normalize_features, MESH_FACE_BREAKS_1000, center_crop_fun, load_image
from src.utils.plot import plot_multi_mesh
from src.utils.solve_pose_multi import solve_pose_multi_obj


def parse_args():
    parser = argparse.ArgumentParser('6D pose estimation demo')

    parser.add_argument('--demo_image_path', type=str, default='imgs/2008_000531.jpg')
    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--model_ckpt_path', type=str, default='models/ckpts')
    parser.add_argument('--mesh_path', type=str, default='models/CAD_single')
    parser.add_argument('--fine_mesh_path', type=str, default='data/PASCAL3D+_release1.1/CAD/car/02.off')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    args.model_ckpt = os.path.join(args.model_ckpt_path, f'{args.category}.pth')
    args.mesh_path = os.path.join(args.mesh_path, args.category)

    return args


def main():
    args = parse_args()
    config = load_default_config()
    config.defrost()
    config.demo_image_path = args.demo_image_path
    config.category = args.category
    config.model_ckpt = args.model_ckpt
    config.mesh_path = args.mesh_path
    config.fine_mesh_path = args.fine_mesh_path
    config.output_dir = args.output_dir
    config.device = args.device
    config.freeze()
    print('\nConfigurations:')
    print(config)

    # prepare model
    net = NetE2E(
        net_type=config.model_parameters.backbone,
        local_size=[config.model_parameters.local_size, config.model_parameters.local_size],
        output_dimension=config.model_parameters.d_features,
        reduce_function=None,
        n_noise_points=0,
        pretrain=True
    )
    net = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(config.model_ckpt, map_location='cpu')
    net.load_state_dict(checkpoint['state'])
    net.eval()
    if isinstance(checkpoint['memory'], torch.Tensor):
        checkpoint['memory'] = [checkpoint['memory']]
    xvert, xface = load_off(os.path.join(config.mesh_path, '01.off'), to_torch=True)
    n = int(xvert.shape[0])
    memory_bank = NearestMemoryManager(inputSize=config.model_parameters.d_features, outputSize=n+0*config.model_parameters.max_group, K=1, num_noise=0,
                                       num_pos=n, momentum=config.model_parameters.adj_momentum)
    memory_bank = memory_bank.cuda()
    with torch.no_grad():
        memory_bank.memory.copy_(checkpoint['memory'][0][0:memory_bank.memory.shape[0]])
    memory = checkpoint['memory'][0][0:memory_bank.memory.shape[0]].detach().cpu().numpy()
    clutter = checkpoint['memory'][0][memory_bank.memory.shape[0]::].detach().cpu().numpy()  # (2560, 128)
    feature_bank = torch.from_numpy(memory)
    clutter_bank = torch.from_numpy(clutter)
    clutter_bank = clutter_bank.cuda()
    clutter_bank = normalize_features(torch.mean(clutter_bank, dim=0)).unsqueeze(0)  # (1, 128)
    kp_features = checkpoint['memory'][0][0:memory_bank.memory.shape[0]].to(config.device)
    clutter_bank = [clutter_bank]

    # prepare renderer
    render_image_size = max(config.image_h, config.image_w) // config.model_parameters.down_sample_rate
    map_shape = (config.image_h//config.model_parameters.down_sample_rate, config.image_w//config.model_parameters.down_sample_rate)
    cameras = PerspectiveCameras(focal_length=12.0, device=config.device)
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=config.rendering_parameters.blur_radius,
        faces_per_pixel=config.rendering_parameters.num_faces,
        bin_size=0
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    inter_module = MeshInterpolateModule(xvert, xface, feature_bank, rasterizer, post_process=center_crop_fun(map_shape, (render_image_size, ) * 2))
    inter_module = inter_module.cuda()

    poses, kp_coords, kp_vis = pre_compute_kp_coords(os.path.join(config.mesh_path, '01.off'),
                                                     mesh_face_breaks=MESH_FACE_BREAKS_1000[config.category],
                                                     azimuth_samples=np.linspace(0, np.pi*2, config.optimization_parameters.azimuth_sample, endpoint=False),
                                                     elevation_samples=np.linspace(-np.pi/6, np.pi/3, config.optimization_parameters.elevation_sample),
                                                     theta_samples=np.linspace(-np.pi/6, np.pi/6, config.optimization_parameters.theta_sample),
                                                     distance_samples=np.linspace(4, 20, config.optimization_parameters.distance_sample, endpoint=True))
    poses = poses.reshape(config.optimization_parameters.azimuth_sample,
                          config.optimization_parameters.elevation_sample,
                          config.optimization_parameters.theta_sample,
                          config.optimization_parameters.distance_sample, 4)

    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    sample_img = load_image(config.demo_image_path, h=config.image_h, w=config.image_w)
    img_tensor = norm(to_tensor(sample_img)).unsqueeze(0)
    with torch.no_grad():
        img_tensor = img_tensor.to(config.device)
        feature_map = net.module.forward_test(img_tensor)

    clutter_score = None
    for cb in clutter_bank:
        cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
        if clutter_score is None:
            clutter_score = cs
        else:
            clutter_score = torch.max(clutter_score, cs)

    pred = solve_pose_multi_obj(
        feature_map, inter_module, kp_features, clutter_bank, poses, kp_coords, kp_vis,
        epochs=config.optimization_parameters.epochs,
        lr=config.optimization_parameters.lr,
        adam_beta_0=config.optimization_parameters.adam_beta_0,
        adam_beta_1=config.optimization_parameters.adam_beta_1,
        mode=config.rendering_parameters.mode,
        device=config.device,
        px_samples=np.linspace(0, config.image_w, config.optimization_parameters.px_sample, endpoint=True),
        py_samples=np.linspace(0, config.image_h, config.optimization_parameters.py_sample, endpoint=True),
        clutter_img_path=None,
        object_img_path=None,
        blur_radius=config.rendering_parameters.blur_radius,
        verbose=False,
        down_sample_rate=config.model_parameters.down_sample_rate
    )
    pred = pred['final']

    vis_mesh_path = os.path.join(config.mesh_path, '01.off') if config.fine_mesh_path is None else config.fine_mesh_path
    img = plot_multi_mesh(sample_img, vis_mesh_path, pred, down_sample_rate=config.model_parameters.down_sample_rate)
    Image.fromarray(img).save(os.path.join(config.output_dir, 'pred.png'))


if __name__ == '__main__':
    main()
