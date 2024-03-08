# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
from glob import glob
import numpy as np
import math
import random
import torch
import cv2, json
from collections import defaultdict
import PIL.Image
from pathlib import Path
from rich.console import Console
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode

CONSOLE = Console(width=180)


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        resolution=64,
        split="train",
        limit_dataset_size=None,
        random_seed=0,
        random_background=False,
        views_per_sample=1,
        verbose=True,
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._name = "PhotoShape"
        self.path = path
        self.has_labels = False
        self.label_shape = None
        self.random_background = random_background

        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder for Photoshape: '{path}' does not exist.")

        self.image_path = Path(os.path.join(path, "straight"))
        self.mask_path = Path(os.path.join(path, "straight_mask"))
        self.pairmeta_path = Path(os.path.join(path, "metadata", "pairs.json"))
        self.shapesmeta_path = Path(os.path.join(path, "metadata", "shapes.json"))

        # load pre-computed mesh id filelist
        self.split = split
        self.file_list = os.path.join(path, "filelist", f"texturify_chair_straight_ids_{split}_v2.txt")
        with open(self.file_list) as f:
            self.obj_names = f.read().splitlines()
        CONSOLE.log(f"load pre-computed mesh id filelist. [split]: {split}")

        # process image and mask
        self.erode = True
        self.real_images_dict = {x.name.split(".")[0]: x for x in self.image_path.iterdir() if x.name.endswith(".jpg") or x.name.endswith(".png")}
        self.real_images_dict = dict(sorted(self.real_images_dict.items()))
        self.masks_dict = {x: self.mask_path / self.real_images_dict[x].name for x in self.real_images_dict}
        self.masks_dict = dict(sorted(self.masks_dict.items()))
        assert self.real_images_dict.keys() == self.masks_dict.keys()
        self.keys_list = list(self.real_images_dict.keys())
        self.num_images = len(self.keys_list)

        self.real_images_preloaded, self.masks_preloaded = {}, {}
        self.pair_meta, self.all_views, self.shape_meta = self.load_pair_meta(self.pairmeta_path, self.shapesmeta_path)
        self.source_id_list = [self.shape_meta[shape_meta_key]["source_id"] for shape_meta_key in self.shape_meta.keys()]

        self.dpsr_chair_path = os.path.join(path, "shapenet_psr", "03001627")
        self.dpsr_obj_names = os.listdir(self.dpsr_chair_path)
        self.raw_shape_meta = json.loads(Path(self.shapesmeta_path).read_text())

        self.items = []
        self.dpsr_items = []
        for obj in self.obj_names:
            shape_id = int(obj.split("_")[0].split("shape")[1])
            shapenet_id = self.raw_shape_meta[str(shape_id)]["source_id"]
            if shapenet_id in self.dpsr_obj_names:
                self.items.append(obj)
                self.dpsr_items.append(shapenet_id)
            else:
                raise ValueError(f"Missing object with shape_id: {shape_id}")

        self.point_cloud_paths = [os.path.join(self.dpsr_chair_path, model, "pointcloud.npz") for model in self.dpsr_items]
        self.num_shapes = len(self.items)

        self.img_size = resolution
        self.views_per_sample = views_per_sample

        # log info
        if verbose:
            CONSOLE.log("==> use image path: %s" % (self.image_path))
            CONSOLE.log("==> num images: %d" % (len(self.real_images_dict)))
            CONSOLE.log("==> num shapes: %d" % (len(self.items)))
            self._raw_shape = [len(self.real_images_dict)] + list(self._load_raw_image(0).shape)

        self._raw_camera_angles = None

        # Apply max_size.
        self._raw_idx = np.arange(self.num_shapes, dtype=np.int64)
        if (limit_dataset_size is not None) and (self._raw_idx.size > limit_dataset_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:limit_dataset_size])

        self.folding_data = None

    def __len__(self):
        return len(self.items)

    def _open_file(self, fname):
        return open(fname, "rb")

    def __getitem__(self, idx):
        # get_image_and_view
        selected_item = self.items[idx]

        shape_id = int(selected_item.split("_")[0].split("shape")[1])

        sampled_view = random.sample(self.all_views, self.views_per_sample)
        image_selections = self.get_image_selections(shape_id)

        BG_COLOR = 255
        if self.random_background:
            if np.random.rand() > 0.5:
                BG_COLOR = 0

        c_i, c_v = image_selections[0], sampled_view[0]
        img = self.get_real_image(self.meta_to_pair(c_i))
        mask = self.get_real_mask(self.meta_to_pair(c_i))

        background = np.ones_like(img) * BG_COLOR  # 255
        img = img * (mask == 0).astype(np.float) + background * (1 - (mask == 0).astype(np.float))

        angles = np.array([c_i["azimuth"] + math.pi, c_i["elevation"], 0, 1.575, c_i["fov"]]).astype(np.float)

        if self.folding_data is None:
            self.__load_folding_packed__()

        # Load point cloud here...
        dense_points = np.load(self.point_cloud_paths[idx])
        surface_points_dense = dense_points["points"].astype(np.float32)
        surface_normals_dense = dense_points["normals"].astype(np.float32)
        pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)

        obj_idx = self.folding_obj_id_dict[selected_item]
        folding_points = self.folding_data["folding_points_all"][obj_idx]
        folding_normals = self.folding_data["folding_normals_all"][obj_idx]
        batch_p_2d = self.folding_data["batch_p_2d_all"][obj_idx]
        folding = np.concatenate([batch_p_2d, folding_points, folding_normals], axis=-1)

        return {
            "obj_idx": obj_idx,
            "image": np.ascontiguousarray(img).astype(np.float32),
            "camera_angles": angles.astype(np.float32),
            "mask": np.ascontiguousarray(mask).astype(np.float32),
            "folding": np.ascontiguousarray(folding).astype(np.float32),  # (2562, 6)
            "pointcloud": np.ascontiguousarray(pointcloud).astype(np.float32),
        }

    def __load_folding_packed__(self):
        packed_path = os.path.join(self.path, "pretrain", "photoshape_foldsdf_v6.npz")
        self.folding_data = np.load(packed_path, allow_pickle=True)
        self.folding_obj_id_dict = {item: i for i, item in enumerate(self.folding_data["texturify_obj_names"])}

    def get_camera_angles(self, idx):
        selected_item = self.items[idx]
        shape_id = int(selected_item.split("_")[0].split("shape")[1])
        image_selections = self.get_image_selections(shape_id)
        sampled_view = random.sample(self.all_views, self.views_per_sample)
        c_i, c_v = image_selections[0], sampled_view[0]
        angles = np.array([c_i["azimuth"] + math.pi, c_i["elevation"], 0, 1.575, c_i["fov"]]).astype(np.float)
        return angles

    def load_pair_meta(self, pairmeta_path, shapesmeta_path):
        loaded_json = json.loads(Path(pairmeta_path).read_text())
        loaded_json_shape = json.loads(Path(shapesmeta_path).read_text())
        ret_shapedict = {}
        ret_dict = defaultdict(list)
        ret_views = []
        for k in loaded_json.keys():
            if self.meta_to_pair(loaded_json[k]) in self.real_images_dict.keys():
                shape_id = loaded_json[k]["shape_id"]
                ret_dict[shape_id].append(loaded_json[k])
                ret_views.append(loaded_json[k])
                ret_shapedict[shape_id] = loaded_json_shape[str(shape_id)]
        return ret_dict, ret_views, ret_shapedict

    def get_image_selections(self, shape_id):
        candidates = self.pair_meta[shape_id]
        if len(candidates) < self.views_per_sample:
            while len(candidates) < self.views_per_sample:
                meta = self.pair_meta[random.choice(list(self.pair_meta.keys()))]
                candidates.extend(meta[: self.views_per_sample - len(candidates)])
        else:
            candidates = random.sample(candidates, self.views_per_sample)
        return candidates

    def get_real_image(self, name):
        if name not in self.real_images_preloaded.keys():
            return self.process_real_image(self.real_images_dict[name])
        else:
            return self.real_images_preloaded[name]

    def get_real_mask(self, name):
        if name not in self.masks_preloaded.keys():
            return self.process_real_mask(self.masks_dict[name])
        else:
            return self.masks_preloaded[name]

    def process_real_image(self, path):
        pad_size = int(self.img_size * 0.1)
        resize = T.Resize(
            size=(self.img_size - 2 * pad_size, self.img_size - 2 * pad_size),
            interpolation=InterpolationMode.BICUBIC,
        )
        pad = T.Pad(padding=(pad_size, pad_size), fill=255)
        t_image = pad(torch.from_numpy(np.array(resize(PIL.Image.open(str(path)))).transpose((2, 0, 1))).float())
        return t_image.numpy()

    def process_real_mask(self, path):
        pad_size = int(self.img_size * 0.1)
        resize = T.Resize(
            size=(self.img_size - 2 * pad_size, self.img_size - 2 * pad_size),
            interpolation=InterpolationMode.NEAREST,
        )
        pad = T.Pad(padding=(pad_size, pad_size), fill=0)
        mask_im = read_image(str(path))[:1, :, :]
        if self.erode:
            eroded_mask = self.erode_mask(mask_im)
        else:
            eroded_mask = mask_im
        t_mask = pad(resize((eroded_mask > 128).float()))
        return (1 - (t_mask[:1, :, :]).float()).numpy()

    @staticmethod
    def erode_mask(mask):
        import cv2 as cv

        mask = mask.squeeze(0).numpy().astype(np.uint8)
        kernel_size = 3
        element = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (2 * kernel_size + 1, 2 * kernel_size + 1),
            (kernel_size, kernel_size),
        )
        mask = cv.erode(mask, element)
        return torch.from_numpy(mask).unsqueeze(0)

    @staticmethod
    def meta_to_pair(c):
        return f'shape{c["shape_id"]:05d}_rank{(c["rank"] - 1):02d}_pair{c["id"]}'

    def _get_raw_camera_angles(self):
        if self._raw_camera_angles is None:
            self._raw_camera_angles = self._load_raw_camera_angles()
            if self._raw_camera_angles is None:
                self._raw_camera_angles = np.zeros([self._raw_shape[0], 3], dtype=np.float32)
            else:
                self._raw_camera_angles = self._raw_camera_angles.astype(np.float32)
            assert isinstance(self._raw_camera_angles, np.ndarray)
            assert self._raw_camera_angles.shape[0] == self._raw_shape[0]
        return self._raw_camera_angles

    def _load_raw_camera_angles(self):
        return None

    def _load_raw_image(self, raw_idx):
        if raw_idx >= len(self.keys_list):
            raise KeyError(raw_idx)

        key = self.keys_list[raw_idx]
        if not os.path.exists(self.real_images_dict[key]):
            raise FileNotFoundError(self.real_images_dict[key])

        img = cv2.imread(str(self.real_images_dict[key]))[..., ::-1]
        img = (img / 255.0).transpose(2, 0, 1)
        return img

    def _load_raw_labels(self):
        return None

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        return self.img_size

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
