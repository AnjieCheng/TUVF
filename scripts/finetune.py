import os
from typing import List
from pathlib import Path

import hydra
import torch
import math, random
import numpy as np
from omegaconf import DictConfig
import torchvision as tv
from torchvision.utils import make_grid
import torchvision.transforms.functional as TVF
from tqdm import tqdm
from torchvision.ops import masks_to_boxes
from cleanfid import fid
from PIL import Image
import pickle

from src.training.inference_utils import generate_camera_angles, save_videos
from scripts.utils import load_generator, set_seed, maybe_makedirs

torch.set_grad_enabled(False)


@hydra.main(config_path="../configs/scripts", config_name="finetune.yaml")
def finetuning(cfg: DictConfig):
    seed = cfg.selected_seed
    device = torch.device("cuda")

    DEMO_DIR = Path(cfg.demo_dir)
    DEMO_DIR_BASE = DEMO_DIR / f"training_views"
    DEMO_DIR_LOGS = DEMO_DIR / f"finetune_logs"
    DEMO_DIR_TEST = DEMO_DIR / f"finetune_test"
    DEMO_DIR_LOGS.mkdir(exist_ok=True, parents=True)
    DEMO_DIR_TEST.mkdir(exist_ok=True, parents=True)

    # ----- prepare camera -----#
    vis_camera = cfg.camera.copy()
    vis_camera["num_frames"] = 180
    vis_trajectory = generate_camera_angles(vis_camera, default_fov=None)[0][:, :2]  # [num_grids, 2]
    # --------------------------#

    # starts here....
    for obj_id in tqdm(range(cfg.start, cfg.end)):
        if obj_id >= cfg.N:
            break

        G = load_generator(cfg, verbose=cfg.verbose)[0].to(device)
        G.synthesis.img_resolution = G.synthesis.train_resolution = G.synthesis.test_resolution = int(1024)  # cfg.img_resolution
        G.synthesis.surface_rendering = False
        G.synthesis.foldsdf_level = 4
        G.synthesis.num_steps_for_render = 1
        G.synthesis.num_steps_for_depth = 32
        G.synthesis.fold_sdf.split = "test"
        G.cfg.num_ray_steps = G.cfg.num_ray_steps * cfg.ray_step_multiplier
        G.cfg.bg_model.num_steps = G.cfg.bg_model.num_steps * cfg.ray_step_multiplier
        G.cfg.dataset.white_back = True if cfg.force_whiteback else G.cfg.dataset.white_back
        G.cfg.dataset.random_background = False
        G.nerf_noise_std = 0

        start_iter = 0
        latent_idx = 0

        LOG_DIR = os.path.join(DEMO_DIR_LOGS, f"geoid-{obj_id:04d}-texseed-{obj_id:04d}")
        maybe_makedirs(LOG_DIR)

        DEMO_DIR_TEST_DIR = os.path.join(DEMO_DIR_TEST, f"geoid-{obj_id:04d}-texseed-{obj_id:04d}")
        maybe_makedirs(DEMO_DIR_TEST_DIR)

        obj_folder = os.path.join(DEMO_DIR_BASE, f"geoid-{obj_id:04d}-texseed-{obj_id:04d}")
        list_of_files = [os.path.join(obj_folder, f"{obj_id:1d}_{c_id:1d}_{latent_idx:1d}.jpg") for c_id in range(cfg.n_views)]

        real_imgs = []
        for file_path in list_of_files:
            img = torch.from_numpy(np.array(Image.open(file_path).resize((G.synthesis.img_resolution, G.synthesis.img_resolution), Image.LANCZOS)).transpose((2, 0, 1))).float()
            real_imgs.append(img)

        real_imgs = torch.stack(real_imgs, dim=0)
        real_imgs = real_imgs.to(device).to(torch.float32) / 127.5 - 1

        seed = obj_id
        set_seed(seed)
        view_list = get_car_views(is_train=True)
        G.synthesis.fold_sdf.test_selected_mode = True
        G.synthesis.fold_sdf.test_selected_idc = seed
        G.synthesis.fold_sdf.split = "test"
        G.synthesis.fold_sdf.check_preload()

        tex_z = sample_z_from_seeds([seed], G.z_dim, device=device).to(device)
        G.synthesis.requires_grad_(True)

        # Initialize optimizer.
        trainable_parameters = list(G.synthesis.parameters())
        optimizer = torch.optim.Adam(trainable_parameters, lr=0.0001)

        train_iters = 20000
        training_views = [3]

        for i in tqdm(range(start_iter, train_iters), ncols=50):
            G.synthesis.train()
            G.synthesis.fold_sdf.test_selected_idc = obj_id
            G.synthesis.img_resolution = G.synthesis.train_resolution = G.synthesis.test_resolution = 1024  # cfg.img_resolution
            G.synthesis.num_steps_for_depth = 96
            with torch.set_grad_enabled(True):
                img_idx = np.random.choice(training_views)
                angles = np.array([view_list[img_idx]["azimuth"], view_list[img_idx]["elevation"], 0, view_list[img_idx]["cam_dist"], view_list[img_idx]["fov"]])[None, :].astype(np.float32)
                angles = torch.from_numpy(angles).float().to(tex_z.device)

                real_img = real_imgs[[img_idx], :, :, :]

                max_batch_res_kwargs = dict(max_batch_res=128)
                synthesis_kwargs = dict(return_depth=False, noise_mode="const", **max_batch_res_kwargs)

                G.synthesis.texture_mlp.requires_grad_(False)
                G.synthesis.local_mlp.requires_grad_(False)

                frame = G.synthesis(tex_z, points=None, camera_angles=angles, fov=None, update_emas=False, return_tex=False, verbose=False, **synthesis_kwargs)
                pred_img = frame[:, :3, :, :]

                loss = torch.nn.functional.mse_loss(pred_img, real_img)
                psnr = mse2psnr(loss.item())

                if i % 100 == 0 or i == train_iters - 1:
                    tqdm.write("[TRAIN] Iter: " + str(i) + " Loss: " + str(loss.item()) + " PSNR: " + str(psnr))

                if i % 500 == 0 or i == train_iters - 1:
                    G.synthesis.eval()
                    with torch.set_grad_enabled(False):
                        TVF.to_pil_image(pred_img[0].clamp(-1, 1).cpu() * 0.5 + 0.5).save(os.path.join(LOG_DIR, f"train_{i}_psnr_{psnr:.3f}.jpg"), q=95)

                if (i % 2500 == 0) or (i == train_iters - 1):
                    G.synthesis.eval()
                    with torch.set_grad_enabled(False):
                        for test_c_id in range(8):
                            test_view_list = get_car_views()
                            test_view_indices = test_c_id
                            # print(view_indices)
                            test_angles = np.array(
                                [
                                    test_view_list[test_view_indices]["azimuth"],
                                    test_view_list[test_view_indices]["elevation"],
                                    0,
                                    test_view_list[test_view_indices]["cam_dist"],
                                    test_view_list[test_view_indices]["fov"],
                                ]
                            ).astype(np.float32)

                            test_grid_camera_angles = torch.from_numpy(test_angles)[None, :].to(device)  # (num_batches, [batch_size, 3])
                            interesting_objects = [21, 7, 28, 45, 46, 25, 11, 30]
                            for test_obj_id in interesting_objects:
                                test_latent_idx = 0
                                G.synthesis.fold_sdf.test_selected_idc = test_obj_id
                                G.synthesis.num_steps_for_depth = 128
                                test_images = G(z=tex_z, camera_angles=test_grid_camera_angles, noise_mode="const").cpu()
                                test_images = bounded_crop(images=test_images, color_space="rgb", resolution=1024)  # [batch_size, c, h, w]

                                for test_image in test_images:
                                    test_image = test_image[:3, :, :]  # .clamp(-1, 1) * 0.5 + 0.5 # [batch_size, c, h, w]
                                    TVF.to_pil_image(test_image).save(os.path.join(DEMO_DIR_TEST_DIR, f"{test_obj_id:1d}_{test_c_id:1d}_{test_latent_idx:1d}.jpg"), q=95)

                if i % 2500 == 0 or i == train_iters - 1:
                    snapshot_modules = [
                        ("G", G),
                    ]
                    print(f"Saving the snapshot...", end="")
                    snapshot_data = {}
                    for name, module in snapshot_modules:
                        snapshot_data[name] = module
                        del module
                    snapshot_pkl = os.path.join(LOG_DIR, f"zckpt_{i:04d}.pkl")
                    with open(snapshot_pkl, "wb") as f:
                        pickle.dump(snapshot_data, f)
                    print("Saved!")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


def get_car_views(is_train=False):
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
    if is_train:
        return [{"azimuth": a + math.pi, "elevation": e, "fov": f, "cam_dist": cd} for a, an, e, en, cd, f in zip(azimuth, azimuth_noise, elevation, elevation_noise, camera_distance, fov)]
    else:
        return [{"azimuth": a + an + math.pi, "elevation": e + en, "fov": f, "cam_dist": cd} for a, an, e, en, cd, f in zip(azimuth, azimuth_noise, elevation, elevation_noise, camera_distance, fov)]


# ----------------------------------------------------------------------------
def bounded_crop(images, color_space="rgb", background=None, resolution=256):
    # import pdb; pdb.set_trace()
    # images: torch.Size([1, 4, 307, 307])
    # preprocess into texturify mesh rendering format
    processed_c = images[:, :3, :, :].clone().permute((0, 2, 3, 1)).contiguous().clamp(-1, 1) * 0.5 + 0.5
    processed_m = 1 - images[:, -1:, :, :].clone().permute((0, 2, 3, 1)).contiguous()
    color = torch.cat([processed_c, processed_m], dim=-1)
    mask = color[..., -1:] == 0
    # import pdb; pdb.set_trace()
    if background is None:
        if color_space == "rgb":
            one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
        else:
            one_tensor = torch.zeros((color.shape[0], color.shape[3], 1, 1), device=color.device)
            one_tensor[:, 0, :, :] = 1
    else:
        one_tensor = background
    # import pdb; pdb.set_trace()
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)  # [:, :, :, :-1]
    color[..., -1:] = mask.float()
    color_crops = []
    boxes = masks_to_boxes(torch.logical_not(mask.squeeze(-1)))
    # import pdb; pdb.set_trace()
    for img_idx in range(color.shape[0]):
        x1, y1, x2, y2 = [int(val) for val in boxes[img_idx, :].tolist()]
        color_crop = color[img_idx, y1:y2, x1:x2, :].permute((2, 0, 1))
        pad = [[0, 0], [0, 0]]
        # import pdb; pdb.set_trace()
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
        # color_crop = T.Pad((pad[0][0], pad[1][0], pad[0][1], pad[1][1]), 1)(color_crop)
        color_crop = torch.nn.functional.interpolate(padded.unsqueeze(0), size=(resolution, resolution), mode="bilinear", align_corners=False).permute((0, 2, 3, 1))
        color_crops.append(color_crop)
        # import pdb; pdb.set_trace()
    return torch.cat(color_crops, dim=0).permute((0, 3, 1, 2)).contiguous()


def sample_z_from_seeds(seeds: List[int], z_dim: int, device: None) -> torch.Tensor:
    zs = [np.random.RandomState(s).randn(1, z_dim) for s in seeds]  # [num_samples, z_dim]
    return torch.from_numpy(np.concatenate(zs, axis=0)).float().to(device)  # [num_samples, z_dim]


def mse2psnr(mse):
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)


if __name__ == "__main__":
    finetuning()

# ----------------------------------------------------------------------------
