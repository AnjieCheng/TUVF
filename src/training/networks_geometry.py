import os
import json
from pathlib import Path
from rich.console import Console
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.utils import ico_sphere

from src.training.networks_dpsr import DPSR
from src.training.networks_spgan import SPGANGenerator, DGCNN

CONSOLE = Console(width=180)

warnings.filterwarnings("ignore")


class SphereTemplate(nn.Module):
    def __init__(self, radius=0.5):
        super().__init__()
        self.dim = 3
        self.radius = radius
        self.ico_sphere = {}
        for i in range(7):
            points = ico_sphere(i)._verts_list[0] * self.radius
            rand_indx = torch.randperm(ico_sphere(i)._verts_list[0].size(0))
            if i == 6:
                self.ico_sphere[i] = points[rand_indx][:40000].float()
            else:
                self.ico_sphere[i] = points[rand_indx].float()

    def get_regular_points(self, level=4, batch_size=None):
        if batch_size is not None:
            points = self.ico_sphere[level][None, :, :]
            points = points.expand(batch_size, -1, -1)
            return points
        else:
            return self.ico_sphere[level]

    def get_random_points(self, num_points=2048, batch_size=None):
        if batch_size is not None:
            rnd = torch.randn(batch_size, num_points, 3, dtype=torch.float)
        else:
            rnd = torch.randn(num_points, 3, dtype=torch.float)
        sphere_samples = (rnd / torch.norm(rnd, dim=-1, keepdim=True)) * self.radius
        return sphere_samples

    def forward(self, points):
        assert points.size(-1) == 3
        points_flat = points.reshape(-1, 3)
        sdfs_flat = torch.linalg.norm(points_flat, dim=-1) - self.radius
        sdfs = sdfs_flat.reshape(points.shape[0], points.shape[1])[..., None]
        return sdfs


class FoldSDF(nn.Module):
    def __init__(
        self,
        feat_dim,
        cfg,
        ckpt_path=None,
        ignore_keys=[],
        name=None,
        split="train",
    ):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = feat_dim

        self.template = SphereTemplate()
        self.Encoder = DGCNN(feat_dim=feat_dim)
        self.Fold_P = SPGANGenerator(z_dim=feat_dim, add_dim=3, use_local=True, use_tanh=True)
        self.Fold_N = SPGANGenerator(z_dim=feat_dim, add_dim=3, use_local=True, use_tanh=False)
        self.dpsr = DPSR(res=(128, 128, 128), sig=4)
        self.split = split

        self.test_selected_mode = False
        self.test_selected_idc = 0

        if "photoshape" in self.cfg.dataset.name:
            self.name = "photoshape"
            ckpt_name = "foldsdf_chair_straight.ckpt"
        elif "compcars" in self.cfg.dataset.name:
            self.name = "compcars"
            ckpt_name = "foldsdf_car.ckpt"
        else:
            raise NotImplementedError

        ckpt_path = os.path.join(self.cfg.dataset.path, "pretrain", ckpt_name)
        self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.is_preloaded = False

    def check_preload(self):
        if not self.is_preloaded:
            self.__preload__()
            self.is_preloaded = True

    def __preload__(self, verbose=True):
        filelist_folder = os.path.join(self.cfg.dataset.path, "filelist")
        if "compcars" in self.cfg.dataset.name:
            self.dpsr_path = os.path.join(self.cfg.dataset.path, "shapenet_psr", "02958343")
            self.filelist_path = os.path.join(filelist_folder, f"texturify_compcars_ids_v2_{self.split}.txt")
            with open(self.filelist_path) as f:
                self.shapenet_obj_names = f.read().splitlines()
            self.texturify_obj_names = self.shapenet_obj_names
            packed_path = os.path.join(self.cfg.dataset.path, "pretrain", "compcars_foldsdf_v4.npz")
            if verbose:
                CONSOLE.log(f"[FoldSDF] loaded CompCars [split]: {self.split}")
                CONSOLE.log(f"[FoldSDF] loaded id filelist for CompCars from: {self.filelist_path}")
                CONSOLE.log(f"[FoldSDF] loaded pretrain packed for CompCars from: {packed_path}")

        elif "photoshape" in self.cfg.dataset.name:
            self.dpsr_path = os.path.join(self.cfg.dataset.path, "shapenet_psr", "03001627")
            self.shapesmeta_path = Path(os.path.join(self.cfg.dataset.path, "metadata", "shapes.json"))
            self.raw_shape_meta = json.loads(Path(self.shapesmeta_path).read_text())
            self.filelist_path = os.path.join(filelist_folder, f"texturify_chair_straight_ids_{self.split}_v2.txt")
            with open(self.filelist_path) as f:
                self.texturify_obj_names = f.read().splitlines()  # texturify format

            self.shapenet_format_filelist_path = os.path.join(filelist_folder, f"shapenet_chair_straight_ids_{self.split}_v2.txt")
            with open(self.shapenet_format_filelist_path) as f:
                self.shapenet_obj_names = f.read().splitlines()  # shapenet format

            packed_path = os.path.join(self.cfg.dataset.path, "pretrain", "photoshape_foldsdf_v6.npz")

            if verbose:
                CONSOLE.log(f"[FoldSDF] loaded Photoshape [split]: {self.split}")
                CONSOLE.log(f"[FoldSDF] loaded Texturify format ids for Photoshape from: {self.filelist_path}")
                CONSOLE.log(f"[FoldSDF] loaded ShapeNet format ids for Photoshape from: {self.shapenet_format_filelist_path}")
                CONSOLE.log(f"[FoldSDF] loaded pretrain packed for Photoshape from: {packed_path}")
        else:
            raise ValueError(f"Missing value with dataset name: {self.cfg.dataset.name}")

        data = np.load(packed_path)

        if "compcars" in self.cfg.dataset.name:
            valid_idc = [i for i, it in enumerate(data["texturify_obj_names"]) if it in self.shapenet_obj_names]
        else:
            valid_idc = [i for i, it in enumerate(data["shapenet_obj_names"]) if it in self.shapenet_obj_names]

        self.point_cloud_paths = [os.path.join(self.dpsr_path, obj_name, "pointcloud.npz") for obj_name in self.shapenet_obj_names]

        self.batch_p_2d_all = data["batch_p_2d_all"][valid_idc]

        self.folding_points_all = data["folding_points_all"][valid_idc]
        self.folding_normals_all = data["folding_normals_all"][valid_idc]
        self.all_points_normals = None

        # we use level 5 ico-sphere for photoshape
        if "dense_folding_points_all" in data:
            self.dense_batch_p_2d_all = data["dense_batch_p_2d_all"][valid_idc]
            self.dense_folding_points_all = data["dense_folding_points_all"][valid_idc]
            self.dense_folding_normals_all = data["dense_folding_normals_all"][valid_idc]
            self.grid_level = 5
            self.contain_dense = True
        else:
            self.grid_level = 4

        self.num_shapes = len(self.batch_p_2d_all)
        self.is_preloaded = True

        if verbose:
            CONSOLE.log(f"FoldSDF pred-loaded with {self.num_shapes} shapes under level {self.grid_level}.")

    def preload(self, batch_size, level, device, rt_gdt_sdf=False, verbose=True):
        self.eval()
        if not self.is_preloaded:
            self.__preload__(verbose=verbose)

        # test mode, just load one instance
        if self.test_selected_mode:
            sub_idc = (np.ones(batch_size) * self.test_selected_idc).astype(np.int64)
        else:
            sub_idc = np.random.choice(self.num_shapes, batch_size)

        rt_batch_p_2d = torch.tensor(self.batch_p_2d_all[sub_idc]).to(device)
        rt_folding_points = torch.tensor(self.folding_points_all[sub_idc]).to(device)
        rt_folding_normals = torch.tensor(self.folding_normals_all[sub_idc]).to(device)

        (
            ts_batch_p_2d,
            ts_folding_points,
            ts_folding_normals,
            rt_sdf_grid,
        ) = self.forward_pred(rt_batch_p_2d, rt_folding_points, rt_folding_normals)

        if self.grid_level > level:
            # we use level 5 ico-sphere for photoshape, overwrite the rt_sdf_grid with highier grid_level
            assert self.contain_dense
            rt_dense_batch_p_2d = torch.tensor(self.dense_batch_p_2d_all[sub_idc]).to(device)
            rt_dense_folding_points = torch.tensor(self.dense_folding_points_all[sub_idc]).to(device)
            rt_dense_folding_normals = torch.tensor(self.dense_folding_normals_all[sub_idc]).to(device)
            _, _, _, rt_sdf_grid = self.forward_pred(rt_dense_batch_p_2d, rt_dense_folding_points, rt_dense_folding_normals)

        batched_pointcloud = None
        if rt_gdt_sdf:
            # overwrite the rt_sdf_grid with gdt sdf
            if self.all_points_normals is not None:
                # deprecated
                rt_sdf_grid = self.forward_gdt(torch.from_numpy(self.all_points_normals[sub_idc]).to(device))
            else:
                # load gdt from raw pointcloud
                assert self.texturify_obj_names is not None
                batched_pointcloud = []
                for idx in sub_idc:
                    dense_points = np.load(self.point_cloud_paths[idx])
                    surface_points_dense = (dense_points["points"]).astype(np.float32)
                    surface_normals_dense = dense_points["normals"].astype(np.float32)
                    pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)
                    batched_pointcloud.append(pointcloud)
                    del pointcloud
                batched_pointcloud = np.stack(batched_pointcloud, axis=0)
                batched_pointcloud = torch.from_numpy(batched_pointcloud).to(device)
                rt_sdf_grid = self.forward_gdt(batched_pointcloud)
        else:
            assert self.texturify_obj_names is not None
            batched_pointcloud = []
            for idx in sub_idc:
                dense_points = np.load(self.point_cloud_paths[idx])
                surface_points_dense = (dense_points["points"]).astype(np.float32)
                surface_normals_dense = dense_points["normals"].astype(np.float32)
                pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)
                batched_pointcloud.append(pointcloud)
                del pointcloud
            batched_pointcloud = np.stack(batched_pointcloud, axis=0)
            batched_pointcloud = torch.from_numpy(batched_pointcloud).to(device)

        return (
            ts_batch_p_2d,
            ts_folding_points,
            rt_folding_normals,
            rt_sdf_grid,
            batched_pointcloud,
        )

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward_gdt(self, points):
        surface_points, surface_normals = points[..., 0:3], points[..., 3:6]
        dpsr_gdt = torch.tanh(self.dpsr(torch.clamp(((surface_points * 0.95) + 0.5), 0.0, 0.99), surface_normals))
        return dpsr_gdt.float()

    def forward_pred(self, batch_p_2d, coords, normals):
        sdf_grid = torch.tanh(self.dpsr(torch.clamp(((coords) + 0.5), 0.0, 0.99).detach(), normals))

        batch_p_2d = batch_p_2d[:, :, torch.LongTensor([2, 1, 0])]
        coords = coords[:, :, torch.LongTensor([2, 1, 0])]
        normals = normals[:, :, torch.LongTensor([2, 1, 0])]

        return batch_p_2d, coords, normals, sdf_grid.float()

    def post_process_sparse_fold(self, folding):
        batch_p_2d = folding[:, :, 0:3].clone()
        coords = folding[:, :, 3:6].clone()
        normals = folding[:, :, 6:9].clone()

        batch_p_2d = batch_p_2d[:, :, torch.LongTensor([2, 1, 0])]
        coords = coords[:, :, torch.LongTensor([2, 1, 0])]
        normals = normals[:, :, torch.LongTensor([2, 1, 0])]

        return batch_p_2d, coords, normals

    def post_process_fold_pred(self, folding):
        batch_p_2d = folding[:, :, 0:3].clone()
        coords = folding[:, :, 3:6].clone()
        normals = folding[:, :, 6:9].clone()

        sdf_grid = torch.tanh(self.dpsr(torch.clamp(((coords) + 0.5), 0.0, 0.99).detach(), normals))

        batch_p_2d = batch_p_2d[:, :, torch.LongTensor([2, 1, 0])]
        coords = coords[:, :, torch.LongTensor([2, 1, 0])]
        normals = normals[:, :, torch.LongTensor([2, 1, 0])]

        return batch_p_2d, coords, normals, sdf_grid

    def forward(self):
        raise NotImplementedError


def sphere_to_color(sphere_coords, radius):
    sphere_coords = np.copy(sphere_coords)
    sphere_coords = np.clip(sphere_coords / (radius * 2) + 0.5, 0, 1)  # normalize color to 0-1
    sphere_coords = np.clip((sphere_coords * 255), 0, 255).astype(int)  # normalize color 0-255
    return sphere_coords
