import os
import numpy as np
import math
import random
import torch
import cv2
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
        random_background=True,
        views_per_sample=1,
        verbose=True,
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._name = "CompCars"
        self.path = path
        self.has_labels = False
        self.label_shape = None
        self.random_background = random_background

        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder for CompCars: '{path}' does not exist.")

        self.image_path = Path(os.path.join(path, "exemplars_highres"))
        self.mask_path = Path(os.path.join(path, "exemplars_highres_mask"))
        self.dpsr_car_path = os.path.join(path, "shapenet_psr", "02958343")

        # process image and mask
        self.erode = True
        self.real_images_dict = {
            x.name.split(".")[0]: x
            for x in self.image_path.iterdir()
            if x.name.endswith(".jpg") or x.name.endswith(".png")
        }
        self.real_images_dict = dict(sorted(self.real_images_dict.items()))
        self.masks_dict = {
            x: self.mask_path / self.real_images_dict[x].name
            for x in self.real_images_dict
        }
        self.masks_dict = dict(sorted(self.masks_dict.items()))
        assert self.real_images_dict.keys() == self.masks_dict.keys()
        self.keys_list = list(self.real_images_dict.keys())
        self.num_images = len(self.keys_list)

        # load mesh id filelist
        self.file_list = os.path.join(
            path, "filelist", f"texturify_compcars_ids_v2_{split}.txt"
        )
        with open(self.file_list) as f:
            self.obj_names = f.read().splitlines()
        CONSOLE.log(f"load pre-computed mesh id filelist. [split]: {split}")

        # load point cloud
        self.point_cloud_paths = [
            os.path.join(self.dpsr_car_path, obj_name, "pointcloud.npz")
            for obj_name in self.obj_names
        ]
        self.num_shapes = len(self.obj_names)

        self.img_size = resolution
        self.views_per_sample = views_per_sample

        # log info
        if verbose:
            CONSOLE.log("==> use image path: %s" % (self.image_path))
            CONSOLE.log("==> num images: %d" % (len(self.real_images_dict)))
            CONSOLE.log("==> num meshes: %d" % (len(self.num_shapes)))

        self._raw_shape = [len(self.real_images_dict)] + list(
            self._load_raw_image(0).shape
        )
        self._raw_camera_angles = None

        # Apply max_size.
        self._raw_idx = np.arange(self.num_shapes, dtype=np.int64)
        if (limit_dataset_size is not None) and (
            self._raw_idx.size > limit_dataset_size
        ):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:limit_dataset_size])

        self.folding_data = None

    def __load_folding_packed__(self):
        # Load folding data
        packed_path = os.path.join(self.path, "pretrain", "compcars_foldsdf_v4.npz")
        self.folding_data = np.load(packed_path, allow_pickle=True)
        self.folding_obj_id_dict = {
            item: i for i, item in enumerate(self.folding_data["texturify_obj_names"])
        }

    def __len__(self):
        return self._raw_idx.size

    def _open_file(self, fname):
        return open(fname, "rb")

    def __getitem__(self, idx):
        # get_image_and_view
        total_selections = len(self.real_images_dict.keys()) // 8
        available_views = get_car_views()
        view_indices = random.sample(list(range(8)), self.views_per_sample)
        sampled_view = [available_views[vidx] for vidx in view_indices]
        image_indices = random.sample(
            list(range(total_selections)), self.views_per_sample
        )
        image_selections = [
            f"{(iidx * 8 + vidx):05d}"
            for (iidx, vidx) in zip(image_indices, view_indices)
        ]

        # get camera position
        angles = np.array(
            [
                sampled_view[0]["azimuth"],
                sampled_view[0]["elevation"],
                0,
                sampled_view[0]["cam_dist"],
                sampled_view[0]["fov"],
            ]
        ).astype(np.float)

        fname = self.real_images_dict[image_selections[0]]
        fname_mask = self.masks_dict[image_selections[0]]

        BG_COLOR = 255
        if self.random_background:
            if np.random.rand() > 0.5:
                BG_COLOR = 0

        img = self.process_real_image(fname)
        mask = self.process_real_mask(fname_mask)
        background = np.ones_like(img) * BG_COLOR  # 255
        img = img * (mask == 0).astype(np.float) + background * (
            1 - (mask == 0).astype(np.float)
        )

        # Load point cloud here...
        selected_obj_id = self.obj_names[idx]
        dense_points = np.load(self.point_cloud_paths[idx])
        surface_points_dense = dense_points["points"].astype(np.float32)
        surface_normals_dense = dense_points["normals"].astype(np.float32)
        pointcloud = np.concatenate(
            [surface_points_dense, surface_normals_dense], axis=-1
        )

        if self.folding_data is None:
            self.__load_folding_packed__()

        # this folding data contains "all" shape datas, need to select only what we need
        selected_obj_index_in_packed_data = self.folding_obj_id_dict[selected_obj_id]
        texturify_obj_name = self.folding_data["texturify_obj_names"][
            selected_obj_index_in_packed_data
        ]
        assert texturify_obj_name == selected_obj_id

        folding_points = self.folding_data["folding_points_all"][
            selected_obj_index_in_packed_data
        ]

        folding_normals = self.folding_data["folding_normals_all"][
            selected_obj_index_in_packed_data
        ]
        batch_p_2d = self.folding_data["batch_p_2d_all"][
            selected_obj_index_in_packed_data
        ]
        folding = np.concatenate([batch_p_2d, folding_points, folding_normals], axis=-1)

        return {
            "image": np.ascontiguousarray(img).astype(np.float32),
            "camera_angles": angles.astype(np.float32),
            "mask": np.ascontiguousarray(mask).astype(np.float32),
            "folding": np.ascontiguousarray(folding).astype(np.float32),
            "pointcloud_id": texturify_obj_name,
            "pointcloud": np.ascontiguousarray(pointcloud).astype(np.float32),
        }

    def process_real_image(self, path):
        pad_size = int(self.img_size * 0.1)
        resize = T.Resize(
            size=(self.img_size - 2 * pad_size, self.img_size - 2 * pad_size),
            interpolation=InterpolationMode.BICUBIC,
        )
        pad = T.Pad(padding=(pad_size, pad_size), fill=255)
        t_image = pad(
            torch.from_numpy(
                np.array(resize(PIL.Image.open(str(path)))).transpose((2, 0, 1))
            ).float()
        )
        return t_image.numpy()

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

    def get_camera_angles(self, idx):
        available_views = get_car_views()
        view_indices = random.sample(list(range(8)), self.views_per_sample)
        sampled_view = [available_views[vidx] for vidx in view_indices]

        # get camera position
        angles = np.array(
            [
                sampled_view[0]["azimuth"],
                sampled_view[0]["elevation"],
                0,
                sampled_view[0]["cam_dist"],
                sampled_view[0]["fov"],
            ]
        ).astype(np.float)
        return angles

    def _get_raw_camera_angles(self):
        if self._raw_camera_angles is None:
            self._raw_camera_angles = self._load_raw_camera_angles()
            if self._raw_camera_angles is None:
                self._raw_camera_angles = np.zeros(
                    [self._raw_shape[0], 3], dtype=np.float32
                )
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


def get_car_views():
    # front, back, right, left, front_right, front_left, back_right, back_left
    camera_distance = [3.2, 3.2, 1.7, 1.7, 1.5, 1.5, 1.5, 1.5]
    fov = [10, 10, 40, 40, 40, 40, 40, 40]
    azimuth = [
        3 * math.pi / 2,
        math.pi / 2,
        0,
        math.pi,
        math.pi + math.pi / 3,
        0 - math.pi / 3,
        math.pi / 2 + math.pi / 6,
        math.pi / 2 - math.pi / 6,
    ]
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
    elevation = [
        math.pi / 2,
        math.pi / 2,
        math.pi / 2,
        math.pi / 2,
        math.pi / 2 - math.pi / 48,
        math.pi / 2 - math.pi / 48,
        math.pi / 2 - math.pi / 48,
        math.pi / 2 - math.pi / 48,
    ]
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
    return [
        {"azimuth": a + an + math.pi, "elevation": e + en, "fov": f, "cam_dist": cd}
        for a, an, e, en, cd, f in zip(
            azimuth, azimuth_noise, elevation, elevation_noise, camera_distance, fov
        )
    ]
