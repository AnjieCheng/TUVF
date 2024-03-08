import numpy as np
import torch
import torch.nn as nn
from skimage import measure


class DPSR(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        super(DPSR, self).__init__()
        self.res = res
        self.sig = sig
        self.dim = len(res)
        self.denom = np.prod(res)
        G = spec_gaussian_filter(res=res, sig=sig).float()
        self.omega = fftfreqs(res, dtype=torch.float32)
        self.scale = scale
        self.shift = shift
        self.register_buffer("G", G)

    def forward(self, V, N):
        assert V.shape == N.shape  # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]

        ras_s = torch.fft.rfftn(ras_p, dim=(2, 3, 4))
        ras_s = ras_s.permute(*tuple([0] + list(range(2, self.dim + 1)) + [self.dim + 1, 1]))
        N_ = ras_s[..., None] * self.G.to(ras_s.device)

        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1)
        omega *= 2 * np.pi
        omega = omega.to(V.device)

        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)

        Lap = -torch.sum(omega**2, -2)  # [dim0, dim1, dim2/2+1, 1]
        Phi = DivN / (Lap + 1e-6)  # [b, dim0, dim1, dim2/2+1, 2]
        Phi = Phi.permute(*tuple([list(range(1, self.dim + 2)) + [0]]))
        Phi[tuple([0] * self.dim)] = 0
        Phi = Phi.permute(*tuple([[self.dim + 1] + list(range(self.dim + 1))]))

        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1, 2, 3))

        if self.shift or self.scale:
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1)  # [b, nv]
            if self.shift:  # offset points to have mean of 0
                offset = torch.mean(fv, dim=-1)  # [b,]
                phi -= offset.view(*tuple([-1] + [1] * self.dim))

            phi = phi.permute(*tuple([list(range(1, self.dim + 1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))

            if self.scale:
                phi = -phi / torch.abs(fv0.view(*tuple([-1] + [1] * self.dim))) * 0.5
        return phi


def fftfreqs(res, dtype=torch.float32, exact=True):
    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1 / r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs, indexing="ij")
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)
    return omega


def img(x, deg=1):
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res


def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=torch.float64)  # [dim0, dim1, dim2, d]
    dis = torch.sqrt(torch.sum(omega**2, dim=-1))
    filter_ = torch.exp(-0.5 * ((sig * 2 * dis / res[0]) ** 2)).unsqueeze(-1).unsqueeze(-1)
    filter_.requires_grad = False

    return filter_


def grid_interp(grid, pts, batched=True):
    if not batched:
        grid = grid.unsqueeze(0)
        pts = pts.unsqueeze(0)
    dim = pts.shape[-1]
    bs = grid.shape[0]
    size = torch.tensor(grid.shape[1:-1]).to(grid.device).type(pts.dtype)
    cubesize = 1.0 / size

    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long()  # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0)  # (2, batch, num_points, dim)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim), indexing="ij"), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1)  # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]  # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)
    if dim == 2:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1]]
    else:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1], ind_n[..., 2]]

    xyz0 = ind0.type(cubesize.dtype) * cubesize  # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0)  # (2, batch, num_points, dim)
    pos_ = xyz01[1 - com_, ..., dim_].permute(2, 3, 0, 1)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize
    weights = torch.prod(dxyz_, dim=-1, keepdim=False)  # (batch, num_points, 2**dim)
    query_values = torch.sum(lat * weights.unsqueeze(-1), dim=-2)
    if not batched:
        query_values = query_values.squeeze(0)

    return query_values


def scatter_to_grid(inds, vals, size):
    dims = inds.shape[1]
    assert inds.shape[0] == vals.shape[0]
    assert len(size) == dims
    dev = vals.device
    result = torch.zeros(*size, device=dev).view(-1).type(vals.dtype)  # flatten
    fac = [np.prod(size[i + 1 :]) for i in range(len(size) - 1)] + [1]
    fac = torch.tensor(fac, device=dev).type(inds.dtype)
    inds_fold = torch.sum(inds * fac, dim=-1)  # [#values,]
    result.scatter_add_(0, inds_fold, vals)
    result = result.view(*size)
    return result


def point_rasterize(pts, vals, size):
    dim = pts.shape[-1]
    assert pts.shape[:2] == vals.shape[:2]
    assert pts.shape[2] == dim
    size_list = list(size)
    size = torch.tensor(size).to(pts.device).float()
    cubesize = 1.0 / size
    bs = pts.shape[0]
    nf = vals.shape[-1]
    npts = pts.shape[1]
    dev = pts.device

    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long()  # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0)  # (2, batch, num_points, dim)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim), indexing="ij"), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1)  # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]  # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    ind_b = torch.arange(bs, device=dev).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)

    xyz0 = ind0.type(cubesize.dtype) * cubesize  # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0)  # (2, batch, num_points, dim)
    pos_ = xyz01[1 - com_, ..., dim_].permute(2, 3, 0, 1)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize
    weights = torch.prod(dxyz_, dim=-1, keepdim=False)  # (batch, num_points, 2**dim)

    ind_b = ind_b.unsqueeze(-1).unsqueeze(-1)  # (batch, num_points, 2**dim, 1, 1)
    ind_n = ind_n.unsqueeze(-2)  # (batch, num_points, 2**dim, 1, dim)
    ind_f = torch.arange(nf, device=dev).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)

    ind_b = ind_b.expand(bs, npts, 2**dim, nf, 1)
    ind_n = ind_n.expand(bs, npts, 2**dim, nf, dim).to(dev)
    ind_f = ind_f.expand(bs, npts, 2**dim, nf, 1)
    inds = torch.cat([ind_b, ind_f, ind_n], dim=-1)

    vals = weights.unsqueeze(-1) * vals.unsqueeze(-2)  # (batch, num_points, 2**dim, nf)

    inds = inds.view(-1, dim + 2).permute(1, 0).long()  # (1+dim+1, bs*npts*2**dim*nf)
    vals = vals.reshape(-1)  # (bs*npts*2**dim*nf)
    raster = scatter_to_grid(inds.permute(1, 0), vals, [bs, nf] + size_list)

    return raster


def mc_from_psr(psr_grid, pytorchify=False, real_scale=False, zero_level=0):
    """
    Run marching cubes from PSR grid
    """
    batch_size = psr_grid.shape[0]
    s = psr_grid.shape[-1]  # size of psr_grid
    psr_grid_numpy = psr_grid.squeeze().detach().cpu().numpy()

    if batch_size > 1:
        verts, faces, normals = [], [], []
        for i in range(batch_size):
            verts_cur, faces_cur, normals_cur, values = measure.marching_cubes(psr_grid_numpy[i], level=0)
            verts.append(verts_cur)
            faces.append(faces_cur)
            normals.append(normals_cur)
        verts = np.stack(verts, axis=0)
        faces = np.stack(faces, axis=0)
        normals = np.stack(normals, axis=0)
    else:
        try:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy, level=zero_level)
        except:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy)
    if real_scale:
        verts = verts / (s - 1)  # scale to range [0, 1]
    else:
        verts = verts / s  # scale to range [0, 1)

    if pytorchify:
        device = psr_grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

    return verts, faces, normals
