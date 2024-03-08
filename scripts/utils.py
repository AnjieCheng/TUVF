import os
import re
import json
import shutil
import random
import itertools
import contextlib
import zipfile
from typing import List, Dict, Tuple

import click
import joblib
from omegaconf import DictConfig
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TVF
from torchvision.utils import make_grid
from tqdm import tqdm
from src import dnnlib, legacy


def create_voxel_coords(resolution=256, voxel_origin=[0.0, 0.0, 0.0], cube_size=2.0, batch_size: int = 1):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_size / 2
    voxel_size = cube_size / (resolution - 1)

    overall_index = torch.arange(0, resolution**3, 1, out=torch.LongTensor())
    coords = torch.zeros(resolution**3, 3)  # [h, w, d, 3]

    # transform first 3 columns
    # to be the x, y, z index
    coords[:, 2] = overall_index % resolution
    coords[:, 1] = (overall_index.float() / resolution) % resolution
    coords[:, 0] = ((overall_index.float() / resolution) / resolution) % resolution

    # transform first 3 columns
    # to be the x, y, z coordinate
    coords[:, 0] = (coords[:, 0] * voxel_size) + voxel_origin[2]  # [voxel_res ** 3]
    coords[:, 1] = (coords[:, 1] * voxel_size) + voxel_origin[1]  # [voxel_res ** 3]
    coords[:, 2] = (coords[:, 2] * voxel_size) + voxel_origin[0]  # [voxel_res ** 3]

    return coords.repeat(batch_size, 1, 1)  # [batch_size, voxel_res ** 3, 3]


# ----------------------------------------------------------------------------


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# ----------------------------------------------------------------------------


def display_dir(dir_path: os.PathLike, num_imgs: int = 25, selection_strategy: str = "order", n_skip_imgs: int = 0, **kwargs) -> "Image":
    Image.init()
    if selection_strategy in ("order", "random"):
        img_fnames = [os.path.relpath(os.path.join(root, fname), start=dir_path) for root, _dirs, files in os.walk(dir_path) for fname in files]
        img_fnames = [f for f in img_fnames if file_ext(f) in Image.EXTENSION]
        img_paths = [os.path.join(dir_path, f) for f in sorted(img_fnames)]
        img_paths = img_paths[n_skip_imgs:]

    if selection_strategy == "order":
        img_paths = img_paths[:num_imgs]
    elif selection_strategy == "random":
        img_paths = random.sample(img_paths, k=num_imgs)
    elif selection_strategy == "random_imgs_from_subdirs":
        img_paths = [p for d in [d for d in listdir_full_paths(dir_path) if os.path.isdir(d)] for p in random.sample(listdir_full_paths(d), k=num_imgs)]
    else:
        raise NotImplementedError(f"Unknown selection strategy: {selection_strategy}")

    return display_imgs(img_paths, **kwargs)


# ----------------------------------------------------------------------------


def display_imgs(img_paths: List[os.PathLike], nrow: bool = None, resize: int = None, crop: Tuple = None, padding: int = 2) -> "Image":
    imgs = [Image.open(p) for p in img_paths]
    if not crop is None:
        imgs = [img.crop(crop) for img in imgs]
    if not resize is None:
        imgs = [TVF.resize(x, size=resize, interpolation=TVF.InterpolationMode.LANCZOS) for x in imgs]
    imgs = torch.stack([TVF.to_tensor(TVF.center_crop(x, output_size=min(x.size))) for x in imgs])  # [num_imgs, c, h, w]
    grid = make_grid(imgs, nrow=(int(np.sqrt(imgs.shape[0])) if nrow is None else nrow), padding=padding)  # [c, grid_h, grid_w]
    grid = TVF.to_pil_image(grid)

    return grid


# ----------------------------------------------------------------------------


def resize_and_save_image(src_path: str, trg_path: str, size: int):
    img = Image.open(src_path)
    img.load()  # required for png.split()
    img = center_resize_crop(img, size)
    jpg_kwargs = {"quality": 95} if file_ext(trg_path) == ".jpg" else {}

    if file_ext(src_path) == ".png" and file_ext(trg_path) == ".jpg" and len(img.split()) == 4:
        jpg = Image.new("RGB", img.size, (255, 255, 255))
        jpg.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        jpg.save(trg_path, **jpg_kwargs)
    else:
        img.save(trg_path, **jpg_kwargs)


# ----------------------------------------------------------------------------


def center_resize_crop(img: Image, size: int) -> Image:
    img = TVF.center_crop(img, min(img.size))  # First, make it square
    img = TVF.resize(img, size, interpolation=TVF.InterpolationMode.LANCZOS)  # Now, resize it

    return img


# ----------------------------------------------------------------------------


def file_ext(path: os.PathLike) -> str:
    return os.path.splitext(path)[1].lower()


# ----------------------------------------------------------------------------


# Extract the zip file for simplicity...
def extract_zip(zip_path: os.PathLike, overwrite: bool = False):
    assert file_ext(zip_path) == ".zip", f"Not a zip archive: {zip_path}"

    if os.path.exists(zip_path[:-4]):
        if overwrite or click.confirm(f"Dir {zip_path[:-4]} already exists. Delete it?", default=False):
            shutil.rmtree(zip_path[:-4])

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path[:-4]))


# ----------------------------------------------------------------------------


def compress_to_zip(dir_to_compress: os.PathLike, delete: bool = False):
    shutil.make_archive(dir_to_compress, "zip", root_dir=os.path.dirname(dir_to_compress), base_dir=os.path.basename(dir_to_compress))

    if delete:
        shutil.rmtree(dir_to_compress)


# ----------------------------------------------------------------------------


def load_generator(raw_cfg: DictConfig, verbose: bool = True, G_key: str = "G_ema") -> Tuple[torch.nn.Module, Dict, str]:
    cfg = raw_cfg.ckpt
    if cfg.network_pkl is None:
        if not cfg.selection_metric is None:
            metrics_file = os.path.join(cfg.networks_dir, f"metric-{cfg.selection_metric}.jsonl")
            with open(metrics_file, "r") as f:
                snapshot_metrics_vals = [json.loads(line) for line in f.read().splitlines()]
            snapshot = sorted(snapshot_metrics_vals, key=lambda m: m["results"][cfg.selection_metric])[0]
            network_pkl = os.path.join(cfg.networks_dir, snapshot["snapshot_pkl"])
            if verbose:
                print(f"Using checkpoint: {network_pkl} with {cfg.selection_metric} of", snapshot["results"][cfg.selection_metric])
        else:
            output_regex = "^network-snapshot-\d{6}.pkl$"
            ckpt_regex = re.compile(output_regex)
            ckpts = sorted([f for f in os.listdir(cfg.networks_dir) if ckpt_regex.match(f)])
            network_pkl = os.path.join(cfg.networks_dir, ckpts[-1])
            if verbose:
                print(f"Using the latest found checkpoint: {network_pkl}")
    else:
        assert cfg.networks_dir is None, "Cant have both parameters: network_pkl and cfg.networks_dir"
        network_pkl = cfg.network_pkl

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        raise ValueError("--network must point to a file or URL")
    if verbose:
        print(f"Loading networks from {network_pkl} with key {G_key}")
    with dnnlib.util.open_url(network_pkl) as f:
        snapshot = legacy.load_network_pkl(f, strict=False)
        if G_key in snapshot:
            G = snapshot[G_key]  # type: ignore
        else:
            print(f"{G_key} not found in {network_pkl}, use key=G instead...)")
            G = snapshot["G"]  # type: ignore

    G.cfg = dnnlib.EasyDict(**G.cfg)
    G.cfg.bg_model = G.cfg.bg_model if "bg_model" in G.cfg else dnnlib.EasyDict(type=None, num_steps=16)
    G.cfg.use_noise = True
    G.cfg.dataset.path = raw_cfg.dataset_path

    G.z_dim = 512

    if cfg.reload_code:
        if cfg.model == "canograf":
            print("Reload code from src.training.networks_canograf...")
            from src.training.networks_canograf import Generator
        else:
            raise NotImplementedError(f"Model {cfg.model} not implemented")
        G_new = Generator(
            G.cfg,
            z_dim=G.z_dim,
            w_dim=G.w_dim,
            img_resolution=G.img_resolution,
            img_channels=G.img_channels,
            c_dim=G.c_dim,
            channel_base=int(G.cfg.get("fmaps", 0.5) * 32768),
            channel_max=G.cfg.get("channel_max", 512),
        )
        G_new.load_state_dict(G.state_dict())
        G = G_new

    return G, snapshot, network_pkl


# ----------------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------


def listdir_full_paths(d: os.PathLike) -> List[os.PathLike]:
    return [os.path.join(d, o) for o in sorted(os.listdir(d))]


# ----------------------------------------------------------------------------


def lanczos_resize_tensors(x: torch.Tensor, size):
    x = [TVF.to_pil_image(img) for img in x]
    x = [TVF.resize(img, size=size, interpolation=TVF.InterpolationMode.LANCZOS) for img in x]
    x = [TVF.to_tensor(img) for img in x]

    return torch.stack(x)


# ----------------------------------------------------------------------------


def maybe_makedirs(d: os.PathLike):
    # TODO: what the hell is this function name?
    if d != "":
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
