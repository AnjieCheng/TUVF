import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

neg = 0.01
neg_2 = 0.2


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim, use_eql=False):
        super().__init__()
        Conv = nn.Conv1d
        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)
        self.style.weight.data.normal_()
        self.style.bias.data.zero_()
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


class EdgeBlock(nn.Module):
    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout // 2, 1),
            nn.BatchNorm2d(Fout // 2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout // 2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.conv_out = nn.Conv2d(Fout, Fout, [1, k], [1, 1])

    def forward(self, x):
        B, C, N = x.shape
        x = get_edge_features(x, self.k)
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w, dim=-1)

        x = self.conv_x(x)
        x = x * w

        x = self.conv_out(x)

        x = x.squeeze(3)

        return x


def get_edge_features(x, k, num=-1, idx=None, return_idx=False):
    B, dims, N = x.shape

    if idx is None:
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt**2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst

        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[:, :, 1 : k + 1]
        idx = idx.contiguous().view(B, N * k)

    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b])
        tmp = tmp.view(dims, N, k).contiguous()
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors)

    # centralize
    central = x.unsqueeze(3)
    central = central.repeat(1, 1, 1, k)

    ee = torch.cat([central, neighbors - central], dim=1)
    assert ee.shape == (B, 2 * dims, N, k)

    if return_idx:
        return ee, idx
    return ee


class SPGANGenerator(nn.Module):
    def __init__(
        self,
        z_dim,
        add_dim=3,
        use_local=True,
        use_tanh=False,
        norm_z=False,
        use_attn=False,
    ):
        super(SPGANGenerator, self).__init__()
        self.np = 2048
        self.nk = 10
        self.nz = z_dim
        self.z_dim = z_dim
        self.off = False
        self.use_attn = False
        self.use_head = False
        self.use_local = use_local
        self.use_tanh = use_tanh
        self.norm_z = norm_z
        self.use_attn = use_attn

        Conv = nn.Conv1d
        Linear = nn.Linear

        dim = 128
        self.head = nn.Sequential(
            Conv(add_dim + self.nz, dim, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim, dim, 1),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.global_conv = nn.Sequential(
            Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(neg, inplace=True),
        )

        if self.use_tanh:
            self.tail_p = nn.Sequential(
                nn.Conv1d(512 + dim, 256, 1),
                nn.LeakyReLU(neg, inplace=True),
                nn.Conv1d(256, 64, 1),
                nn.LeakyReLU(neg, inplace=True),
                nn.Conv1d(64, 3, 1),
                nn.Tanh(),
            )
        else:
            self.tail_p = nn.Sequential(
                nn.Conv1d(512 + dim, 256, 1),
                nn.LeakyReLU(neg, inplace=True),
                nn.Conv1d(256, 64, 1),
                nn.LeakyReLU(neg, inplace=True),
                nn.Conv1d(64, 3, 1),
            )

        if self.use_head:
            self.pc_head = nn.Sequential(
                Conv(add_dim, dim // 2, 1),
                nn.LeakyReLU(inplace=True),
                Conv(dim // 2, dim, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.EdgeConv1 = EdgeBlock(dim, dim, self.nk)
            self.adain1 = AdaptivePointNorm(dim, dim)
            self.EdgeConv2 = EdgeBlock(dim, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)
        else:
            self.EdgeConv1 = EdgeBlock(add_dim, 64, self.nk)
            self.adain1 = AdaptivePointNorm(64, dim)
            self.EdgeConv2 = EdgeBlock(64, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)

        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)

    def forward(self, z, x):
        B, N, _ = x.size()
        if self.norm_z:
            z = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1).contiguous()
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1).contiguous()
        feat_global = feat_global.repeat(1, 1, N)

        if self.use_local:
            feat_cat = torch.cat((feat_global, x2), dim=1)
        else:
            feat_cat = feat_global

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        x1_p = self.tail_p(feat_cat).transpose(1, 2).contiguous()  # Bx3x256

        if self.use_tanh:
            x1_p = x1_p / 2

        return x1_p


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1).contiguous() * num_points
    else:
        idx_base = (
            torch.arange(0, batch_size, device=idx.get_device())
            .view(-1, 1, 1)
            .contiguous()
            * num_points
        )
    idx = idx + idx_base
    idx = idx.view(-1).contiguous()

    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points).contiguous()
    if idx is None:
        idx = knn(x, k=k)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


class DGCNN(nn.Module):
    def __init__(self, feat_dim):
        super(DGCNN, self).__init__()
        self.k = 20

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(feat_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, feat_dim, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.global_conv = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        local_x = self.conv6(x)

        global_x = local_x.max(dim=-1, keepdim=False)[0]

        global_x = local_x.max(dim=-1, keepdim=True)[0]
        global_conv_x = self.global_conv(global_x.squeeze(dim=-1))

        return global_conv_x, local_x.permute(0, 2, 1).contiguous()
