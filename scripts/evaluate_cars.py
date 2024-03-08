import os
import random
import math
import warnings
import numpy as np
import torch
from typing import List
from pathlib import Path
from PIL import Image
from omegaconf import DictConfig
from rich.console import Console
import hydra

import torchvision.transforms.functional as TVF
from tqdm import tqdm
from torchvision.ops import masks_to_boxes
from cleanfid import fid

from scripts.utils import load_generator, set_seed

CONSOLE = Console(width=180)
warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)


@hydra.main(config_path="../configs/scripts", config_name="evaluate.yaml", version_base="1.3")
def create_real(cfg: DictConfig):
    save_dir = cfg.output_dir

    REAL_DIR = Path(os.path.join(save_dir, "real"))
    REAL_DIR.mkdir(exist_ok=True, parents=True)

    image_path = Path(os.path.join(cfg.dataset_path, "exemplars_highres"))
    mask_path = Path(os.path.join(cfg.dataset_path, "exemplars_highres_mask"))

    CONSOLE.print("Processing Real Images...")

    image_size = 256
    for idx, p in enumerate(tqdm(list(image_path.iterdir()))):
        if idx == 5000:
            break
        img = Image.open(p)
        width, height = img.size
        result = Image.new(img.mode, (int(width * 1.2), int(height * 1.2)), (255, 255, 255))
        result.paste(img, (int(width * 0.1), int(height * 0.1)))
        result = result.resize((image_size, image_size), resample=Image.LANCZOS)
        mask = np.array(Image.open(mask_path / p.name))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = Image.fromarray(mask)
        width, height = mask.size
        mask_result = Image.new(mask.mode, (int(width * 1.2), int(height * 1.2)), 0)
        mask_result.paste(mask, (int(width * 0.1), int(height * 0.1)))
        mask_result = mask_result.resize((image_size, image_size), resample=Image.NEAREST)
        mask_arr = (np.array(mask_result) > 128).astype(np.uint8)
        result_arr = (np.array(result) * mask_arr[:, :, None] + np.ones_like(np.array(result)) * 255 * (1 - mask_arr[:, :, None])).astype(np.uint8)

        result = Image.fromarray(result_arr)
        result.save(REAL_DIR / f"{idx:06d}.jpg")
    CONSOLE.rule("Finished creating real images.")


@hydra.main(config_path="../configs/scripts", config_name="evaluate.yaml", version_base="1.3")
def generate_samples(cfg: DictConfig):
    device = torch.device("cuda")
    save_dir = cfg.output_dir
    set_seed(cfg.seed)  # To fix non-z randomization

    REAL_DIR = Path(os.path.join(save_dir, "real"))
    OUTPUT_DIR = Path(os.path.join(save_dir, "generated_samples_fid_kid"))
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    G = load_generator(cfg, verbose=cfg.verbose)[0].to(device).eval()
    G.synthesis.img_resolution = G.synthesis.test_resolution = int(512 * 1.2)  # cfg.img_resolution
    G.synthesis.num_steps_for_render = 3
    G.synthesis.num_steps_for_depth = 128
    G.cfg.num_ray_steps = G.cfg.num_ray_steps * cfg.ray_step_multiplier
    G.cfg.bg_model.num_steps = G.cfg.bg_model.num_steps * cfg.ray_step_multiplier
    G.cfg.dataset.white_back = True if cfg.force_whiteback else G.cfg.dataset.white_back
    G.nerf_noise_std = 0

    batch_size = 1

    G.synthesis.fold_sdf.test_selected_mode = True
    G.synthesis.fold_sdf.split = "test"
    G.synthesis.fold_sdf.check_preload()

    CONSOLE.print("Generating Images...")

    with torch.no_grad():
        n_latent = 4  # number of latent
        view_list = get_car_views()

        for obj_id in tqdm(range(G.synthesis.fold_sdf.num_shapes)):
            G.synthesis.fold_sdf.test_selected_idc = obj_id

            for c_id in range(4):
                view_indices = random.choice(list(range(8)))
                angles = np.array([view_list[view_indices]["azimuth"], view_list[view_indices]["elevation"], 0, view_list[view_indices]["cam_dist"], view_list[view_indices]["fov"]]).astype(np.float32)

                grid_camera_angles = torch.from_numpy(angles)[None, :].expand(batch_size, -1).to(device)

                for latent_idx in range(n_latent):
                    grid_z = torch.randn([batch_size, G.z_dim], device=device)
                    images = G(z=grid_z, camera_angles=grid_camera_angles, noise_mode="const").cpu()
                    images = torch.nn.functional.interpolate(images, (256, 256), mode="bilinear", align_corners=True)

                    for image in images:
                        image = image[:3, :, :].clamp(-1, 1) * 0.5 + 0.5  # [batch_size, c, h, w]
                        TVF.to_pil_image(image).save(os.path.join(OUTPUT_DIR, f"{obj_id:1d}_{c_id:1d}_{latent_idx:1d}.jpg"), q=95)

    FID_SCORE = fid.compute_fid(str(REAL_DIR), str(OUTPUT_DIR), device=device, dataset_res=256, num_workers=8)
    CONSOLE.log(f"FID: {FID_SCORE:.5f}")
    KID_SCORE = fid.compute_kid(str(REAL_DIR), str(OUTPUT_DIR), device=device, dataset_res=256, num_workers=8)
    CONSOLE.log(f"KID: {KID_SCORE:.5f}")

    filepath = Path(OUTPUT_DIR) / "score.txt"
    Path(filepath).write_text(f"FID: {FID_SCORE:.5f} KID: {KID_SCORE:.5f}")
    CONSOLE.rule("Finished calculating FID/KID.")


# ----------------------------------------------------------------------------


@hydra.main(config_path="../configs/scripts", config_name="evaluate.yaml", version_base="1.3")
def evaluate_disentangle_fixed_z(cfg: DictConfig):
    seed = cfg.selected_seed
    device = torch.device("cuda")

    save_dir = cfg.output_dir
    OUTPUT_DIR = Path(os.path.join(save_dir, "generated_samples_disentangle_fixed_z"))
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    G = load_generator(cfg, verbose=cfg.verbose)[0].to(device).eval()
    G.synthesis.img_resolution = G.synthesis.test_resolution = int(512 * 1.2)
    G.cfg.num_ray_steps = G.cfg.num_ray_steps * cfg.ray_step_multiplier
    G.cfg.bg_model.num_steps = G.cfg.bg_model.num_steps * cfg.ray_step_multiplier
    G.cfg.dataset.white_back = True if cfg.force_whiteback else G.cfg.dataset.white_back
    G.nerf_noise_std = 0
    G.synthesis.fold_sdf.split = "test"
    G.synthesis.fold_sdf.test_selected_mode = True
    G.synthesis.fold_sdf.check_preload()

    batch_size = 1
    N = 100
    seeds = [0, 1, 2, 3]
    views = [0, 1, 2]
    view_list = get_car_views()

    OUTPUT_DIR_SEED_list = []

    CONSOLE.print("Generating Disentangle_fixed_z Samples...")

    for seed in seeds:
        set_seed(seed)
        grid_z = torch.randn([1, G.z_dim], device=device)

        for view in views:
            OUTPUT_DIR_SEED = OUTPUT_DIR / f"seed_{seed}_view_{view}"
            OUTPUT_DIR_SEED.mkdir(exist_ok=True, parents=True)

            for obj_id in tqdm(range(N)):
                G.synthesis.fold_sdf.test_selected_idc = obj_id
                angles = np.array([view_list[view]["azimuth"], view_list[view]["elevation"], 0, view_list[view]["cam_dist"], view_list[view]["fov"]]).astype(np.float32)
                grid_camera_angles = torch.from_numpy(angles)[None, :].expand(batch_size, -1).to(device)

                with torch.no_grad():
                    image = G(z=grid_z, camera_angles=grid_camera_angles, noise_mode="const").cpu()

                image = bounded_crop(images=image, color_space="rgb", resolution=512)
                image = torch.nn.functional.interpolate(image, (256, 256), mode="bilinear", align_corners=True)[0]
                image = image[:3, :, :]

                img_path = OUTPUT_DIR_SEED / f"{obj_id}_{view}.jpg"
                TVF.to_pil_image(image).save(img_path, q=95)
                del image

            OUTPUT_DIR_SEED_list.append(OUTPUT_DIR_SEED)

    del G
    avg_dist_list = []
    for OUTPUT_DIR_SEED in OUTPUT_DIR_SEED_list:
        from src.metrics.lpips import compute_lpips

        avg_dist = compute_lpips(OUTPUT_DIR_SEED, OUTPUT_DIR_SEED.parent / f"score{N}_allpairs_{seed}_{view}.txt", N=None, all_pairs=True, use_gpu=True)
        avg_dist_list.append(avg_dist)

    LPIPS_T_SCORE = np.mean(avg_dist_list)
    CONSOLE.log(f"LPIPS_t: {LPIPS_T_SCORE:.5f}")
    CONSOLE.rule("Finished calculating LPIPS_t.")


@hydra.main(config_path="../configs/scripts", config_name="evaluate.yaml", version_base="1.3")
def evaluate_disentangle_fixed_geo(cfg: DictConfig):
    device = torch.device("cuda")

    save_dir = cfg.output_dir
    OUTPUT_DIR = Path(os.path.join(save_dir, "generated_samples_disentangle_fixed_geo"))
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    G = load_generator(cfg, verbose=cfg.verbose)[0].to(device).eval()
    G.synthesis.img_resolution = G.synthesis.test_resolution = int(512 * 1.2)
    G.cfg.num_ray_steps = G.cfg.num_ray_steps * cfg.ray_step_multiplier
    G.cfg.bg_model.num_steps = G.cfg.bg_model.num_steps * cfg.ray_step_multiplier
    G.cfg.dataset.white_back = True if cfg.force_whiteback else G.cfg.dataset.white_back
    G.nerf_noise_std = 0
    G.synthesis.fold_sdf.split = "test"
    G.synthesis.fold_sdf.test_selected_mode = True
    G.synthesis.fold_sdf.check_preload()

    batch_size = 1
    num_latent = 10
    views = [0, 1, 2, 3]
    N = 100
    view_list = get_car_views()

    OUTPUT_DIR_SEED_list = []

    CONSOLE.print("Generating Disentangle_fixed_geo Samples...")

    for view in views:
        for obj_id in tqdm(range(N)):
            G.synthesis.fold_sdf.test_selected_idc = obj_id
            angles = np.array([view_list[view]["azimuth"], view_list[view]["elevation"], 0, view_list[view]["cam_dist"], view_list[view]["fov"]]).astype(np.float32)
            grid_camera_angles = torch.from_numpy(angles)[None, :].expand(batch_size, -1).to(device)

            OUTPUT_DIR_SEED = OUTPUT_DIR / f"obj_{obj_id}_view_{view}"
            OUTPUT_DIR_SEED.mkdir(exist_ok=True, parents=True)

            for z_idx in range(num_latent):
                grid_z = torch.randn([1, G.z_dim], device=device)

                with torch.no_grad():
                    image = G(z=grid_z, camera_angles=grid_camera_angles, noise_mode="const").cpu()

                image = bounded_crop(images=image, color_space="rgb", resolution=512)
                image = torch.nn.functional.interpolate(image, (256, 256), mode="bilinear", align_corners=True)[0]

                image = image[:3, :, :]

                img_path = OUTPUT_DIR_SEED / f"{z_idx}.jpg"
                TVF.to_pil_image(image).save(img_path, q=95)
                del image

            OUTPUT_DIR_SEED_list.append(OUTPUT_DIR_SEED)

    del G
    avg_dist_list = []
    for OUTPUT_DIR_SEED in OUTPUT_DIR_SEED_list:
        from src.metrics.lpips import compute_lpips

        avg_dist = compute_lpips(OUTPUT_DIR_SEED, OUTPUT_DIR_SEED.parent / f"score_lpips_{N}_allpairs_{obj_id}_{view}.txt", N=None, all_pairs=True, use_gpu=True)
        avg_dist_list.append(avg_dist)

    LPIPS_G_SCORE = np.mean(avg_dist_list)
    CONSOLE.log(f"LPIPS_g: {LPIPS_G_SCORE:.5f}")
    CONSOLE.rule("Finished calculating LPIPS_g.")


# ----------------------------------------------------------------------------


def sample_z_from_seeds(seeds: List[int], z_dim: int) -> torch.Tensor:
    zs = [np.random.RandomState(s).randn(1, z_dim) for s in seeds]  # [num_samples, z_dim]
    return torch.from_numpy(np.concatenate(zs, axis=0)).float()  # [num_samples, z_dim]


# ----------------------------------------------------------------------------


def get_car_views():
    # front, back, right, left, front_right, front_left, back_right, back_left
    camera_distance = [3.2, 3.2, 1.7, 1.7, 1.5, 1.5, 1.5, 1.5]
    fov = [10, 10, 40, 40, 40, 40, 40, 40]
    azimuth = [3 * math.pi / 2, math.pi / 2, 0, math.pi, math.pi + math.pi / 3, 0 - math.pi / 3, math.pi / 2 + math.pi / 6, math.pi / 2 - math.pi / 6]
    azimuth_noise = [
        0,
        0,
        0,
        0,
        (random.random() - 0.5) * math.pi / 7,
        (random.random() - 0.5) * math.pi / 7,
        (random.random() - 0.5) * math.pi / 7,
        (random.random() - 0.5) * math.pi / 7,
    ]
    elevation = [math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48]
    elevation_noise = [
        -random.random() * math.pi / 32,
        -random.random() * math.pi / 32,
        0,
        0,
        -random.random() * math.pi / 28,
        -random.random() * math.pi / 28,
        -random.random() * math.pi / 28,
        -random.random() * math.pi / 28,
    ]
    return [{"azimuth": a + an + math.pi, "elevation": e + en, "fov": f, "cam_dist": cd} for a, an, e, en, cd, f in zip(azimuth, azimuth_noise, elevation, elevation_noise, camera_distance, fov)]


# ----------------------------------------------------------------------------
def bounded_crop(images, color_space="rgb", background=None, resolution=256):
    processed_c = images[:, :3, :, :].clone().permute((0, 2, 3, 1)).contiguous().clamp(-1, 1) * 0.5 + 0.5
    processed_m = 1 - images[:, -1:, :, :].clone().permute((0, 2, 3, 1)).contiguous()
    color = torch.cat([processed_c, processed_m], dim=-1)
    mask = color[..., -1:] == 0
    if background is None:
        if color_space == "rgb":
            one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
        else:
            one_tensor = torch.zeros((color.shape[0], color.shape[3], 1, 1), device=color.device)
            one_tensor[:, 0, :, :] = 1
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)  # [:, :, :, :-1]
    color[..., -1:] = mask.float()
    color_crops = []
    boxes = masks_to_boxes(torch.logical_not(mask.squeeze(-1)))
    for img_idx in range(color.shape[0]):
        x1, y1, x2, y2 = [int(val) for val in boxes[img_idx, :].tolist()]
        color_crop = color[img_idx, y1:y2, x1:x2, :].permute((2, 0, 1))
        pad = [[0, 0], [0, 0]]
        if y2 - y1 > x2 - x1:
            total_pad = (y2 - y1) - (x2 - x1)
            pad[0][0] = total_pad // 2
            pad[0][1] = total_pad - pad[0][0]
            pad[1][0], pad[1][1] = 0, 0
            additional_pad = int((y2 - y1) * 0.1)
        else:
            total_pad = (x2 - x1) - (y2 - y1)
            pad[0][0], pad[0][1] = 0, 0
            pad[1][0] = total_pad // 2
            pad[1][1] = total_pad - pad[1][0]
            additional_pad = int((x2 - x1) * 0.1)
        for i in range(4):
            pad[i // 2][i % 2] += additional_pad

        padded = torch.ones((color_crop.shape[0], color_crop.shape[1] + pad[1][0] + pad[1][1], color_crop.shape[2] + pad[0][0] + pad[0][1]), device=color_crop.device)
        padded[:3, :, :] = padded[:3, :, :] * one_tensor[img_idx, :3, :, :]
        padded[:, pad[1][0] : padded.shape[1] - pad[1][1], pad[0][0] : padded.shape[2] - pad[0][1]] = color_crop
        color_crop = torch.nn.functional.interpolate(padded.unsqueeze(0), size=(resolution, resolution), mode="bilinear", align_corners=False).permute((0, 2, 3, 1))
        color_crops.append(color_crop)
    return torch.cat(color_crops, dim=0).permute((0, 3, 1, 2)).contiguous()


if __name__ == "__main__":
    create_real()
    generate_samples()
    evaluate_disentangle_fixed_geo()
    evaluate_disentangle_fixed_z()
# ----------------------------------------------------------------------------
