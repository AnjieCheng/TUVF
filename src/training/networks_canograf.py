from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.torch_utils import misc
from src.torch_utils import persistence
from omegaconf import DictConfig

from pytorch3d.ops import knn_points
from src.training.networks_geometry import FoldSDF
from src.training.networks_cips import CIPSres
from src.training.layers import (
    FullyConnectedLayer,
    ScalarEncoder1d,
)
from src.training.training_utils import batched_index_select, run_batchwise
from src.training.rendering import (
    fancy_integration,
    get_initial_rays_trig,
    transform_points,
    sample_pdf,
    compute_cam2world_matrix,
    get_depth_z,
    extract_density_from_sdf_grid,
)


@misc.profiled_function
def canonical_renderer_pretrain(
    uv_x: torch.Tensor,
    coords: torch.Tensor,
    sdf_grid: torch.Tensor,
    folding_coords: torch.Tensor,
    texture_mlp: Callable,
    local_mlp: Callable,
    beta: torch.Tensor,
) -> torch.Tensor:
    # geo
    batch_size, _, _ = uv_x.shape
    num_points = coords.shape[1]

    coords_normed = coords / 0.5
    sdfs = F.grid_sample((sdf_grid - 100), coords_normed.view(batch_size, 1, 1, num_points, 3), padding_mode="zeros", align_corners=False).view(batch_size, num_points, 1) + 100
    sigmas = (1 / beta) * (0.5 + 0.5 * (sdfs).sign() * torch.expm1(-(sdfs).abs() / beta))

    K = 4
    dis, indices, _ = knn_points(coords.detach(), folding_coords.detach(), K=K)
    weights = 1 / (dis.sqrt() + 1e-7)
    weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)

    k_query_folding_feats = batched_index_select(uv_x, 1, indices).view(batch_size, num_points, K, 32)
    k_query_folding_points = batched_index_select(folding_coords, 1, indices).view(batch_size, num_points, K, 3)
    k_query_points_expanded = coords[:, :, None, :].expand(-1, -1, K, -1)
    k_query_local_coords = k_query_points_expanded - k_query_folding_points

    k_query_folding_feats_transformed = local_mlp(k_query_folding_feats, k_query_local_coords)
    fused_feats = torch.sum(k_query_folding_feats_transformed.view(batch_size, num_points, K, 32) * weights[..., None], dim=-2)
    rgbs = texture_mlp(fused_feats)

    """
    ---- uncomment this part to see correspondence visualization ---- 
    K = 1
    dis, indices, _ = knn_points(coords.detach(), folding_coords.detach(), K=K)
    normed_bp2d = torch.clip((folding_grid+0.5), 0, 1) # normalize color to 0-1
    rgbs = rgbs * 0 + (batched_index_select(normed_bp2d, 1, indices).view(batch_size, num_points, 3))
    """

    return torch.cat([rgbs, sigmas.detach()], dim=-1)


# ----------------------------------------------------------------------------
@persistence.persistent_class
class LocalMLP(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_enc = ScalarEncoder1d(3, x_multiplier=self.cfg.texture.posenc_period_len, const_emb_dim=0)
        backbone_input_dim = self.pos_enc.get_dim() + 32
        backbone_out_dim = 32
        self.dims = [backbone_input_dim] + [self.cfg.texture.mlp.hid_dim] * (self.cfg.texture.mlp.n_layers - 1) + [backbone_out_dim]  # (n_hid_layers + 2)
        activations = ["lrelu"] * (len(self.dims) - 2) + ["linear"]
        assert len(self.dims) > 2, f"We cant have just a linear layer here: nothing to modulate. Dims: {self.dims}"
        layers = [FullyConnectedLayer(self.dims[i], self.dims[i + 1], activation=a) for i, a in enumerate(activations)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, local_coords: torch.Tensor = None) -> torch.Tensor:
        batch_size, num_points, K, feat_dim = x.shape
        x = x.reshape(batch_size * num_points * K, feat_dim)
        local_coords = local_coords.reshape(batch_size * num_points * K, 3)

        local_coords = self.pos_enc(local_coords)
        x = torch.cat([x, local_coords], dim=1)

        x = self.model(x)
        x = x.view(batch_size, num_points, K, self.dims[-1])
        return x


@persistence.persistent_class
class TextureMLP(nn.Module):
    def __init__(self, cfg: DictConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim

        has_view_cond = self.cfg.texture.view_hid_dim > 0

        if self.cfg.texture.mlp.n_layers == 0:
            assert self.cfg.texture.feat_dim == (self.out_dim + 1), f"Wrong dims: {self.cfg.texture.feat_dim}, {self.out_dim}"
            self.model = nn.Identity()
        else:
            if self.cfg.texture.get("posenc_period_len", 0) > 0:
                self.pos_enc = ScalarEncoder1d(32, x_multiplier=self.cfg.texture.posenc_period_len, const_emb_dim=0)
            else:
                self.pos_enc = None
            self.pos_enc = None

            backbone_input_dim = 32
            backbone_out_dim = self.cfg.texture.mlp.hid_dim if has_view_cond else self.out_dim
            self.dims = [backbone_input_dim] + [self.cfg.texture.mlp.hid_dim] * (self.cfg.texture.mlp.n_layers - 1) + [backbone_out_dim]  # (n_hid_layers + 2)
            activations = ["lrelu"] * (len(self.dims) - 2) + ["linear"]
            assert len(self.dims) > 2, f"We cant have just a linear layer here: nothing to modulate. Dims: {self.dims}"
            layers = [FullyConnectedLayer(self.dims[i], self.dims[i + 1], activation=a) for i, a in enumerate(activations)]
            self.model = nn.Sequential(*layers)

            if self.cfg.texture.view_hid_dim > 0:
                self.ray_dir_enc = ScalarEncoder1d(coord_dim=3, const_emb_dim=0, x_multiplier=64, use_cos=False)
                self.color_network = nn.Sequential(
                    FullyConnectedLayer(
                        self.cfg.texture.view_hid_dim + self.ray_dir_enc.get_dim(),
                        self.cfg.texture.view_hid_dim,
                        activation="lrelu",
                    ),
                    FullyConnectedLayer(self.cfg.texture.view_hid_dim, self.out_dim, activation="linear"),
                )
            else:
                self.ray_dir_enc = None
                self.color_network = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, feat_dim = x.shape
        x = x.reshape(batch_size * num_points, feat_dim)
        x = self.model(x)
        y = x.view(batch_size, num_points, self.dims[-1])
        return y


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        cfg: DictConfig,  # Main config
        z_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        num_fp16_res=4,  # Number of FP16 res blocks for the upsampler
        **synthesis_seq_kwargs,  # Arguments of SynthesisBlocksSequence / deprecated
    ):
        super().__init__()
        self.cfg = cfg
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.fold_sdf = FoldSDF(feat_dim=256, ignore_keys=["dpsr"], cfg=self.cfg)
        self.fold_sdf.eval()

        # rgb
        self.cfg.texture.type = "cips"
        if self.cfg.texture.type == "cips":
            self.texture_decoder = CIPSres(style_dim=512, out_dim=32)
        else:
            raise NotImplementedError(f"Unknown texture type: {self.cfg.texture.type}")

        self.texture_mlp = TextureMLP(self.cfg, out_dim=3)
        self.local_mlp = LocalMLP(self.cfg)

        self.beta = 0.005  # 0.005 # nn.Parameter(torch.tensor(0.005))

        self.num_ws = self.texture_decoder.num_ws
        self.nerf_noise_std = 0.0
        self.train_resolution = self.cfg.patch.resolution if self.cfg.patch.enabled else self.img_resolution
        self.test_resolution = self.img_resolution if self.img_resolution <= 256 else 256
        self.surface_rendering = False
        self.faster_volumetric_rendering = True
        self.use_preload_sdf_to_train = True
        self.use_gdt_sdf = False
        self.cropped = False

        self.foldsdf_level = 4
        self.num_steps_for_depth = 256
        self.num_steps_for_render = 3

        if self.cfg.bg_model.type in (None, "plane"):
            self.bg_model = None
        else:
            raise NotImplementedError(f"Uknown BG model type: {self.bg_model}")

    def progressive_update(self, cur_kimg: float):
        self.nerf_noise_std = 0

    def forward(
        self,
        tex_z,
        camera_angles,
        points=None,
        sdfs=None,
        f=None,
        patch_params=None,
        max_batch_res=128,
        return_depth=False,
        ignore_bg=False,
        bg_only=False,
        fov=None,
        verbose=False,
        return_tex=False,
        **block_kwargs,
    ):
        # init some rendering params
        rgb_sigma_out_dim = 4
        foldsdf_level = self.foldsdf_level
        num_steps = self.num_steps_for_depth
        nerf_noise_std = 0.0

        if camera_angles.size(1) == 3:
            radius = self.cfg.dataset.sampling.radius
        elif camera_angles.size(1) == 5:
            camera_angles = camera_angles.float()
            radius = camera_angles[:, 3]
            fov = camera_angles[:, 4]
            camera_angles = camera_angles[:, :3]

        batch_size = tex_z.shape[0]
        h = w = self.train_resolution if self.training else self.test_resolution
        fov = self.cfg.dataset.sampling.fov if fov is None else fov  # [1] or [batch_size]

        # apply white background
        bg_color = torch.ones((batch_size, 1, 1)).to(tex_z.device)
        if "random_background" in self.cfg.dataset:
            # apply random background
            if self.training and self.cfg.dataset.random_background:
                bg_color = torch.randint(2, (batch_size, 1, 1)).to(tex_z.device)

        # sanity check for empty inputs
        with torch.no_grad():
            if points is not None:
                if torch.isnan(points).any():
                    points = None
                    f = None

            """
            ####### probably for finetuning only ########
            batch_p_2d, folding_points, folding_normals, sdf_grid_pred, points = self.fold_sdf.preload(batch_size, level=foldsdf_level, rt_gdt_sdf=False, device=tex_z.device)
            sdf_grid = sdf_grid_pred.view(batch_size, 1, *self.fold_sdf.dpsr.res)
            #############################################
            """
            batch_p_2d = None
            if self.training or points is None:  # training & eval go here
                batch_p_2d, folding_points, folding_normals, sdf_grid_pred, _ = self.fold_sdf.preload(batch_size, level=foldsdf_level, rt_gdt_sdf=False, device=tex_z.device)
                sdf_grid = sdf_grid_pred.view(batch_size, 1, *self.fold_sdf.dpsr.res)

            else:  # vis goes here
                if points is not None and f is not None:
                    points = points.to(tex_z.device)
                    folding = f.to(tex_z.device)

                    if folding.shape[1] > 10242:
                        # We use dense points for Photoshape
                        sparse_folding = folding[:, 0:2562, :]
                        dense_folding = folding[:, 2562:, :]
                        batch_p_2d, folding_points, folding_normals = self.fold_sdf.post_process_sparse_fold(sparse_folding)

                        # get sdf_grid from dense points
                        _, _, _, sdf_grid = self.fold_sdf.post_process_fold_pred(dense_folding)

                    else:
                        # We use sparse points for CompCars
                        batch_p_2d, folding_points, folding_normals, sdf_grid = self.fold_sdf.post_process_fold_pred(folding)
                    sdf_grid = sdf_grid.view(batch_size, 1, *self.fold_sdf.dpsr.res)

        if sdfs is not None:
            sdf_grid = sdfs

        # generate uv features
        uv_feats = self.texture_decoder(batch_p_2d, [tex_z])  # .squeeze(3).transpose(1,2) # [batch_size, feat_dim, tp_h, tp_w]

        # sample rays
        z_vals, rays_d_cam = get_initial_rays_trig(
            batch_size,
            num_steps,
            resolution=(h, w),
            device=tex_z.device,
            ray_start=self.cfg.dataset.sampling.ray_start,
            ray_end=self.cfg.dataset.sampling.ray_end,
            fov=fov,
            patch_params=patch_params,
            radius=radius,
        )
        c2w = compute_cam2world_matrix(camera_angles, radius)
        points_world, z_vals, ray_d_world, ray_o_world = transform_points(z_vals=z_vals, ray_directions=rays_d_cam, c2w=c2w)
        points_world = points_world.reshape(batch_size, h * w * num_steps, 3)

        # extract density from sdf grid
        density_output = run_batchwise(
            fn=extract_density_from_sdf_grid,
            data=dict(coords=points_world),
            batch_size=max_batch_res**2 * num_steps,
            dim=1,
            sdf_grid=sdf_grid,
            beta=self.beta,
        )
        density_output = density_output.view(batch_size, h * w, num_steps, 1)

        # get depth for each pixel
        coarse_geo = run_batchwise(
            fn=get_depth_z,
            data=dict(sigmas=density_output, z_vals=z_vals),
            batch_size=max_batch_res**2,
            dim=1,
            clamp_mode=self.cfg.clamp_mode,
            use_inf_depth=self.cfg.bg_model.type is None,
        )
        weights = coarse_geo["weights"].reshape(batch_size * h * w, num_steps) + 1e-5

        # importance sampling base on depth
        z_vals = z_vals.reshape(batch_size * h * w, num_steps)  # [batch_size * h * w, num_steps]
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # [batch_size * h * w, num_steps - 1]
        z_vals = z_vals.reshape(batch_size, h * w, num_steps, 1)  # [batch_size, h * w, num_steps, 1]

        num_steps = 3  # [3 or 50]
        fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
        fine_z_vals = fine_z_vals.reshape(batch_size, h * w, num_steps, 1)

        fine_points = ray_o_world.unsqueeze(2).contiguous() + ray_d_world.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
        fine_points = fine_points.reshape(batch_size, h * w * num_steps, 3)

        # rgb prediction on re-sampled fined-points
        fine_output = run_batchwise(
            fn=canonical_renderer_pretrain,
            data=dict(coords=fine_points),
            batch_size=max_batch_res**2 * num_steps,
            dim=1,
            texture_mlp=self.texture_mlp,
            local_mlp=self.local_mlp,
            uv_x=uv_feats,
            folding_coords=folding_points,
            sdf_grid=sdf_grid,
            beta=self.beta,
        )
        fine_output = fine_output.view(batch_size, h * w, num_steps, rgb_sigma_out_dim)

        fine_rgb_sigma = fine_output[..., :rgb_sigma_out_dim]
        fine_points = fine_points.reshape(batch_size, h * w, num_steps, 3)

        # Sort by z_values
        _, indices = torch.sort(fine_z_vals, dim=2)
        all_z_vals = torch.gather(fine_z_vals, dim=2, index=indices)
        all_rgb_sigma = torch.gather(
            fine_rgb_sigma,
            dim=2,
            index=indices.expand(-1, -1, -1, rgb_sigma_out_dim),
        )

        int_out: Dict = run_batchwise(
            fn=fancy_integration,
            data=dict(rgb_sigma=all_rgb_sigma, z_vals=all_z_vals),
            batch_size=max_batch_res**2,
            dim=1,
            bg_color=bg_color,
            last_back=self.cfg.dataset.last_back,
            clamp_mode=self.cfg.clamp_mode,
            noise_std=nerf_noise_std,
            use_inf_depth=self.cfg.bg_model.type is None,
        )
        misc.assert_shape(int_out["final_transmittance"], [batch_size, h * w])

        img = int_out["depth" if return_depth else "rendered_feats"]
        img = img.reshape(batch_size, h, w, img.shape[2])
        img = img.permute(0, 3, 1, 2).contiguous()

        mask = coarse_geo["final_transmittance"].reshape(batch_size, h, w, 1).permute(0, 3, 1, 2).contiguous()

        img = torch.cat([img, mask], dim=1)

        if verbose:
            info = {}
            return img, info
        else:
            if return_tex:
                return img, None
            else:
                return img


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self, cfg, z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs={}, **synthesis_kwargs):
        super().__init__()
        self.cfg = cfg
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(cfg=cfg, z_dim=self.z_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)

    def progressive_update(self, cur_kimg: float):
        self.synthesis.progressive_update(cur_kimg)

    def forward(self, z, camera_angles, p=None, f=None, sdfs=None, update_emas=False, warp_folding=False, **synthesis_kwargs):
        img = self.synthesis(z, points=p, f=f, sdfs=sdfs, camera_angles=camera_angles, update_emas=update_emas, warp_folding=warp_folding, **synthesis_kwargs)
        return img
