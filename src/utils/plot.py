import numpy as np
from pytorch3d.renderer import OpenGLPerspectiveCameras, RasterizationSettings, MeshRasterizer, MeshRenderer, PointLights, HardPhongShader, PerspectiveCameras
from pytorch3d.renderer import TexturesVertex as Textures
from pytorch3d.structures import Meshes
import torch

from .mesh import load_off, pre_process_mesh_pascal, campos_to_R_T, camera_position_from_spherical_angles


def plot_mesh(img, mesh_path, azimuth, elevation, theta, distance, principal, down_sample_rate=8, fuse=True):
    h, w, c = img.shape
    render_image_size = max(h, w)
    crop_size = (h, w)

    # cameras = OpenGLPerspectiveCameras(device='cuda:0', fov=12.0)
    cameras = PerspectiveCameras(focal_length=12.0, device='cuda:0')
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    raster_settings1 = RasterizationSettings(
        image_size=render_image_size // down_sample_rate,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings1
    )
    lights = PointLights(device='cuda:0', location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device='cuda:0', lights=lights, cameras=cameras)
    )

    x3d, xface = load_off(mesh_path)
    verts = torch.from_numpy(x3d).to('cuda:0')
    verts = pre_process_mesh_pascal(verts)
    faces = torch.from_numpy(xface).to('cuda:0')
    verts_rgb = torch.ones_like(verts)[None]
    verts_rgb = torch.ones_like(verts)[None] * torch.Tensor([1, 0.85, 0.85]).view(1, 1, 3).to(verts.device)
    textures = Textures(verts_rgb.to('cuda:0'))
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures)

    C = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=False, device='cuda:0')
    R, T = campos_to_R_T(C, theta, device='cuda:0')
    image = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
    image = image[:, ..., :3]

    # box_ = bbt.box_by_shape(crop_size, (render_image_size // 2,) * 2)
    # bbox = box_.bbox
    # image = image[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :]
    l = render_image_size//2 - crop_size[0]//2
    t = render_image_size//2 - crop_size[1]//2
    image = image[:, l:l+crop_size[0], t:t+crop_size[1], :]

    image = torch.squeeze(image).detach().cpu().numpy()
    image = np.array((image / image.max()) * 255).astype(np.uint8)

    # cy, cx = principal
    # dx = int(- cx + h/2)
    # dy = int(- cy + w/2)
    cx, cy = principal
    dx = int(-cx + w/2)
    dy = int(-cy + h/2)
    # image = np.roll(image, int(-dx), axis=0)
    # image = np.roll(image, int(-dy), axis=1)
    image_pad = np.pad(image, ((abs(dy), abs(dy)), (abs(dx), abs(dx)), (0, 0)), mode='edge')
    image = image_pad[dy+abs(dy):dy+abs(dy)+image.shape[0], dx+abs(dx):dx+abs(dx)+image.shape[1]]

    a = 0.8
    mask = (image.sum(2) != 765)[:, :, np.newaxis]
    img = img * (1 - a * mask) + image * a * mask
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)

    if fuse:
        get_image = alpha_merge_imgs(img, image)
        # get_image = np.concatenate([raw_img, image], axis=1)
        return get_image
    else:
        return image


def get_nocs_features(xvert):
    xvert = xvert.clone()
    xvert[:, 0] -= (torch.min(xvert[:, 0]) + torch.max(xvert[:, 0]))/2.0
    xvert[:, 1] -= (torch.min(xvert[:, 1]) + torch.max(xvert[:, 1]))/2.0
    xvert[:, 2] -= (torch.min(xvert[:, 2]) + torch.max(xvert[:, 2]))/2.0
    xvert[:, 0] /= torch.max(torch.abs(xvert[:, 0])) * 2.0
    xvert[:, 1] /= torch.max(torch.abs(xvert[:, 1])) * 2.0
    xvert[:, 2] /= torch.max(torch.abs(xvert[:, 2])) * 2.0
    xvert += 0.5
    return xvert


def keypoint_score(feature_map, memory, nocs, clutter_score=None, device='cuda:0'):
    if not torch.is_tensor(feature_map):
        feature_map = torch.tensor(feature_map, device=device).unsqueeze(0) # (1, C, H, W)
    if not torch.is_tensor(memory):
        memory = torch.tensor(memory, device=device) # (nkpt, C)

    nkpt, c = memory.size()
    feature_map = feature_map.expand(nkpt, -1, -1, -1)
    memory = memory.view(nkpt, c, 1, 1)

    kpt_map = torch.sum(feature_map * memory, dim=1) # (nkpt, H, W)
    kpt_map, idx = torch.max(kpt_map, dim=0)

    nocs_map = nocs[idx, :].view(kpt_map.shape[0], kpt_map.shape[1], 3).to(device)

    nocs_map = nocs_map * kpt_map.unsqueeze(2)

    if clutter_score is not None:
        nocs_map[kpt_map < clutter_score] = 0.0

    return nocs_map.detach().cpu().numpy()


def plot_multi_mesh(img, mesh_path, pred, down_sample_rate=8, fuse=True):
    img_list = [plot_mesh(img, mesh_path, p['azimuth'], p['elevation'], p['theta'], p['distance'], p['principal'], down_sample_rate, fuse=fuse) for p in pred]
    s = img_list[0]
    m = np.sum(img_list[0], axis=2) < 255*3
    for i in range(1, len(img_list)):
        mi = (np.sum(img_list[i], axis=2) < 255*3) & (~m)
        s[mi, :] = img_list[i][mi, :]
        m = m | mi
    if fuse:
        # get_image = alpha_merge_imgs(img, s)
        # return get_image
        a = 0.7
        mask = (s.sum(2) != 765)[:, :, np.newaxis]
        img = img * (1 - a * mask) + s * a * mask
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)
    else:
        return s
