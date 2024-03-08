from typing import Tuple, Dict, Callable, Any, List
import torch
import torch.nn.functional as F
import numpy as np

from pytorch3d.ops import knn_points
from torchvision.ops import masks_to_boxes


# preprocess into texturify mesh rendering format
def bounded_crop(images, color_space="rgb", background=None, resolution=256):
    processed_c = images[:, :3, :, :].clone().permute((0, 2, 3, 1)).contiguous()  # .clamp(-1, 1) * 0.5 + 0.5
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


def sample_sphere_points(radius=0.3, num_points=4096):
    rnd = torch.randn(num_points, 3, dtype=torch.float)
    sphere_samples = (rnd / torch.norm(rnd, dim=-1, keepdim=True)) * radius
    return sphere_samples


def get_feat_from_triplane(coords, feature_triplane, scale=None):
    batch_size, raw_feat_dim, h, w = feature_triplane.shape
    num_points = coords.shape[1]
    feat_dim = raw_feat_dim // 3

    feature_triplane = feature_triplane.view(batch_size * 3, feat_dim, h, w)  # [batch_size * 3, feat_dim, h, w]
    if scale is not None:
        coords = coords / scale  # [batch_size, num_points, 3]
    coords_2d = torch.stack(
        [
            coords[..., [0, 1]],  # x/y plane
            coords[..., [0, 2]],  # x/z plane
            coords[..., [1, 2]],  # y/z plane
        ],
        dim=1,
    )  # [batch_size, 3, num_points, 2]
    coords_2d = coords_2d.view(batch_size * 3, 1, num_points, 2)  # [batch_size * 3, 1, num_points, 2]
    # assert ((coords_2d.min().item() >= -1.0 - 1e-8) and (coords_2d.max().item() <= 1.0 + 1e-8))
    coord_feat = F.grid_sample(feature_triplane, grid=coords_2d, mode="bilinear", align_corners=True).view(batch_size, 3, feat_dim, num_points)  # [batch_size, 3, feat_dim, num_points]
    coord_feat = coord_feat.permute(0, 1, 3, 2)  # [batch_size, 3, num_points, feat_dim]
    return coord_feat


def get_nn_fused_feats(coords, folding_points, folding_feat, K=8):
    feat_dim = folding_feat.shape[-1]
    batch_size, num_points, _ = coords.shape
    dis, indices, _ = knn_points(coords, folding_points, K=K, return_nn=False)
    dis = dis.detach()
    indices = indices.detach()
    dis = dis.sqrt()
    weights = 1 / (dis + 1e-7)
    weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)
    folding_feat = folding_feat.transpose(1, 2)  # (B, num_folding_points, 3, 32)
    nn_features = batched_index_select(folding_feat, 1, indices).reshape(batch_size, num_points, K, 3, feat_dim)
    fused_tex_feature = torch.sum(nn_features * weights[..., None, None], dim=-3)
    return fused_tex_feature.transpose(1, 2)  # [batch_size, 3, num_points, feat_dim]


def forward_uniform(
    geo_ws: torch.Tensor,
    tex_x: torch.Tensor,
    geo_x: torch.Tensor,
    coords: torch.Tensor,
    ray_d_world: torch.Tensor,
    mlp_f: Callable,
    mlp_b: Callable,
    template: Callable,
    texture_mlp: Callable,
    scale: float = 1.0,
    rt_sdf: bool = False,
) -> torch.Tensor:
    # geo
    # with torch.no_grad():
    assert geo_x.shape[1] % 3 == 0, f"We use 3 planes: {geo_x.shape}"
    batch_size, raw_feat_dim, h, w = geo_x.shape

    geo_feat = get_feat_from_triplane(coords, geo_x, scale=scale)

    # if geo_x.requires_grad:
    # with torch.set_grad_enabled(True):
    # coords_org = coords.clone().detach().requires_grad_(True)
    # coords = coords_org
    coords_f = F.tanh(mlp_f(geo_feat, coords, ray_d_world, geo_ws))  # [batch_size, num_points, out_dim]
    # sigmas, sdfs = template(coords_f, get_sdf=True)
    # sdf_grad = gradient(sdfs, coords)

    geo_feat_f = get_feat_from_triplane(coords_f, geo_x, scale=None)
    coords_b = F.tanh(mlp_b(geo_feat_f, coords_f, ray_d_world, geo_ws))  # * scale # F.tanh() * scale

    return torch.cat([coords, coords_f, coords_b], dim=-1)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    return grad


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


# ----------------------------------------------------------------------------


def linear_schedule(step: int, val_start: float, val_end: float, period: int) -> float:
    """
    Returns the current value from `val_from` to `val_to` if the linear growth over `period` steps is used
    If out of range, then returns the boundary value
    """
    if step >= period:
        return val_end
    elif step <= 0:
        return val_start
    else:
        return val_start + (val_end - val_start) * step / period


# ----------------------------------------------------------------------------


def extract_patches(x: torch.Tensor, patch_params: Dict, resolution: int) -> torch.Tensor:
    """
    Extracts patches from images and interpolates them to a desired resolution
    Assumes, that scales/offests in patch_params are given for the [0, 1] image range (i.e. not [-1, 1])
    """
    _, _, h, w = x.shape
    assert h == w, "Can only work on square images (for now)"
    coords = compute_patch_coords(patch_params, resolution)  # [batch_size, resolution, resolution, 2]
    out = F.grid_sample(x, coords, mode="bilinear", align_corners=True)  # [batch_size, c, resolution, resolution]
    return out


# ----------------------------------------------------------------------------


def compute_patch_coords(patch_params: Dict, resolution: int, align_corners: bool = True, for_grid_sample: bool = True) -> torch.Tensor:
    """
    Given patch parameters and the target resolution, it extracts
    """
    patch_scales, patch_offsets = patch_params["scales"], patch_params["offsets"]  # [batch_size, 2], [batch_size, 2]
    batch_size, _ = patch_scales.shape
    coords = generate_coords(batch_size=batch_size, img_size=resolution, device=patch_scales.device, align_corners=align_corners)  # [batch_size, out_h, out_w, 2]

    # First, shift the coordinates from the [-1, 1] range into [0, 2]
    # Then, multiply by the patch scales
    # After that, shift back to [-1, 1]
    # Finally, apply the offset converted from [0, 1] to [0, 2]
    coords = (coords + 1.0) * patch_scales.view(batch_size, 1, 1, 2) - 1.0 + patch_offsets.view(batch_size, 1, 1, 2) * 2.0  # [batch_size, out_h, out_w, 2]

    if for_grid_sample:
        # Transforming the coords to the layout of `F.grid_sample`
        coords[:, :, :, 1] = -coords[:, :, :, 1]  # [batch_size, out_h, out_w]

    return coords


# ----------------------------------------------------------------------------


def sample_patch_params(batch_size: int, patch_cfg: Dict, device: str = "cpu") -> Dict:
    """
    Samples patch parameters: {scales: [x, y], offsets: [x, y]}
    It assumes to follow image memory layout
    """
    if patch_cfg["distribution"] == "uniform":
        return sample_patch_params_uniform(
            batch_size=batch_size,
            min_scale=patch_cfg["min_scale"],
            max_scale=patch_cfg["max_scale"],
            group_size=patch_cfg["mbstd_group_size"],
            device=device,
        )
    elif patch_cfg["distribution"] == "discrete_uniform":
        return sample_patch_params_uniform(
            batch_size=batch_size,
            min_scale=patch_cfg["min_scale"],
            max_scale=patch_cfg["max_scale"],
            discrete_support=patch_cfg["discrete_support"],
            group_size=patch_cfg["mbstd_group_size"],
            device=device,
        )
    elif patch_cfg["distribution"] == "beta":
        return sample_patch_params_beta(
            batch_size=batch_size,
            min_scale=patch_cfg["min_scale"],
            max_scale=patch_cfg["max_scale"],
            alpha=patch_cfg["alpha"],
            beta=patch_cfg["beta"],
            group_size=patch_cfg["mbstd_group_size"],
            device=device,
        )
    else:
        raise NotImplementedError(f'Unkown patch sampling distrubtion: {patch_cfg["distribution"]}')


# ----------------------------------------------------------------------------


def sample_patch_params_uniform(batch_size: int, min_scale: float, max_scale: float, discrete_support: List[float] = None, group_size: int = 1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales and patch offsets
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
    """
    assert max_scale <= 1.0, f"Too large max_scale: {max_scale}"
    assert min_scale <= max_scale, f"Incorrect params: min_scale = {min_scale}, max_scale = {max_scale}"

    num_groups = batch_size // group_size

    if discrete_support is None:
        patch_scales_x = np.random.rand(num_groups) * (max_scale - min_scale) + min_scale  # [num_groups]
    else:
        # Sampling from the discrete distribution
        curr_support = [s for s in discrete_support if min_scale <= s <= max_scale]
        patch_scales_x = np.random.choice(curr_support, size=num_groups, replace=True).astype(np.float32)  # [num_groups]

    return create_patch_params_from_x_scales(patch_scales_x, group_size, **kwargs)


# ----------------------------------------------------------------------------


def sample_patch_params_beta(batch_size: int, min_scale: float, max_scale: float, alpha: float, beta: float, group_size: int = 1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales and patch offsets
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
    """
    assert max_scale <= 1.0, f"Too large max_scale: {max_scale}"
    assert min_scale <= max_scale, f"Incorrect params: min_scale = {min_scale}, max_scale = {max_scale}"
    num_groups = batch_size // group_size
    patch_scales_x = np.random.beta(a=alpha, b=beta, size=num_groups) * (max_scale - min_scale) + min_scale
    return create_patch_params_from_x_scales(patch_scales_x, group_size, **kwargs)


# ----------------------------------------------------------------------------


def create_patch_params_from_x_scales(patch_scales_x: np.ndarray, group_size: int = 1, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Since we assume that patches are square and we sample assets uniformly,
    we can share a lot of code parts.
    """
    patch_scales_x = torch.from_numpy(patch_scales_x).float().to(device)
    patch_scales = torch.stack([patch_scales_x, patch_scales_x], dim=1)  # [num_groups, 2]

    # Sample an offset from [0, 1 - patch_size]
    patch_offsets = torch.rand(patch_scales.shape, device=device) * (1.0 - patch_scales)  # [num_groups, 2]

    # Replicate the groups (needed for the MiniBatchStdLayer)
    patch_scales = patch_scales.repeat_interleave(group_size, dim=0)  # [batch_size, 2]
    patch_offsets = patch_offsets.repeat_interleave(group_size, dim=0)  # [batch_size, 2]

    return {"scales": patch_scales, "offsets": patch_offsets}


# ----------------------------------------------------------------------------


def generate_coords(batch_size: int, img_size: int, device="cpu", align_corners: bool = False) -> torch.Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[idx, 0, 0] = (-1, 1)
    - lower right corner: coords[idx, -1, -1] = (1, -1)
    In this way, the `y` axis is flipped to follow image memory layout
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float()  # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1  # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1)  # [img_size, img_size]
    y_coords = -x_coords.t()  # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2)  # [img_size, img_size, 2]
    coords = coords.view(-1, 2)  # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1)  # [batch_size, 2, img_size, img_size]
    coords = coords.permute(0, 2, 3, 1)  # [batch_size, 2, img_size, img_size]

    return coords


# ----------------------------------------------------------------------------


def run_batchwise(fn: Callable, data: Dict[str, torch.Tensor], batch_size: int, dim: int = 0, **kwargs) -> Any:
    """
    Runs a function in a batchwise fashion along the `dim` dimension to prevent OOM
    Params:
        - fn: the function to run
        - data: a dict of tensors which should be split batchwise
    """
    # Filter out None data types
    keys, values = zip(*data.items())
    assert batch_size >= 1, f"Wrong batch_size: {batch_size}"
    assert len(set([v.shape[dim] for v in values])) == 1, f"Tensors must be of the same size along dimension {dim}. Got {[v.shape[dim] for v in values]}"

    # Early exit
    if values[0].shape[dim] <= batch_size:
        return fn(**data, **kwargs)

    results = []
    num_runs = (values[0].shape[dim] + batch_size - 1) // batch_size

    for i in range(num_runs):
        assert dim == 1, f"Sorry, works only for dim=1, while provided dim={dim}"
        curr_data = {k: d[:, i * batch_size : (i + 1) * batch_size] for k, d in data.items()}
        results.append(fn(**curr_data, **kwargs))

    if isinstance(results[0], torch.Tensor):
        return torch.cat(results, dim=dim)
    elif isinstance(results[0], list) or isinstance(results[0], tuple):
        return [torch.cat([r[i] for r in results], dim=dim) for i in range(len(results[0]))]
    elif isinstance(results[0], dict):
        return {k: torch.cat([r[k] for r in results], dim=dim) for k in results[0].keys()}
    else:
        raise NotImplementedError(f"Cannot handle {type(results[0])} result types.")


# ----------------------------------------------------------------------------
