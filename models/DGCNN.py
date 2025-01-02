#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils.distance import knn
from models.MLP import Rotation_fix_axis_mirror
from utils.rotate import get_rotation_matrix


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(1)
        # x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        knn_idx = idx
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, _, num_dims = x.size()

        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.contiguous().view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return knn_idx, feature

    def forward(self, x):
        # batch_size = x.size(0)
        _, x = self.get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = x1.transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        x = torch.cat((x1, x2, x3, x4), dim=-1)

        return x


class DGCNN_3layers(nn.Module):
    def __init__(self, args):
        super(DGCNN_3layers, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(1)
        # x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        knn_idx = idx
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, _, num_dims = x.size()

        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.contiguous().view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return knn_idx, feature

    def forward(self, x):
        # batch_size = x.size(0)
        _, x = self.get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = x1.transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        x = torch.cat((x1, x2, x3), dim=-1)

        return x


class DGCNN_4layers(nn.Module):
    def __init__(self, args):
        super(DGCNN_4layers, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(1)
        # x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        knn_idx = idx
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, _, num_dims = x.size()

        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.contiguous().view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return knn_idx, feature

    def forward(self, x):
        # batch_size = x.size(0)
        idx, x = self.get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = x1.transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x1, k=self.k, idx=idx)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x2, k=self.k, idx=idx)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        x = torch.cat((x1, x2, x3, x4), dim=-1)

        return x


class DGCNN_simple(nn.Module):
    def __init__(self, args):
        super(DGCNN_simple, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
        #                            self.bn3,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(1)
        # x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        knn_idx = idx
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, _, num_dims = x.size()

        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.contiguous().view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return knn_idx, feature

    def forward(self, x):
        # batch_size = x.size(0)
        idx, x = self.get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = x1.transpose(2, 1).contiguous()

        _, x = self.get_graph_feature(x1, k=self.k, idx=idx)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        # _, x = get_graph_feature(x2, k=self.k, idx=idx)
        # x = self.conv3(x)
        # x3 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1)

        _, x = self.get_graph_feature(x2, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        x = torch.cat((x1, x2, x4), dim=-1)

        return x


class DGCNN_3layers_rotate(nn.Module):
    def __init__(self, args):
        super(DGCNN_3layers_rotate, self).__init__()
        self.args = args
        self.k = args.k
        self.rotate_mirror = Rotation_fix_axis_mirror()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(9, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

    def get_graph_feature(self, x, normals=None, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(1)
        num_dims = x.size(2)
        R_matrix1 = None
        R_matrix2 = None
        # x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        knn_idx = idx
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.contiguous().view(batch_size * num_points, -1)[idx, :]
        if num_dims == 3:
            feature = feature.view(-1, k, 3)
            x = x.view(-1, 1, 3).repeat(1, k, 1)
            feature = feature - x
            direction = torch.FloatTensor([0, 0, 1]).repeat(batch_size * num_points, 1).to(device)
            normals = normals.view(-1, 1, 3)
            R_matrix1 = get_rotation_matrix(normals, direction)
            rotate_f = torch.bmm(R_matrix1, feature.transpose(2, 1).contiguous()).transpose(2 ,1).contiguous()

            feature, R_matrix2 = self.rotate_mirror(rotate_f, direction)

            feature = feature.view(batch_size, num_points, k, 2 * num_dims)
            feature = torch.cat((feature, x.view(batch_size, num_points, k, 3)), dim=3)
            feature = feature.permute(0, 3, 1, 2).contiguous()
        else:
            feature = feature.view(batch_size, num_points, k, num_dims)
            x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
            feature = torch.cat((feature - x, x), dim=3)
            feature = feature.permute(0, 3, 1, 2).contiguous()

        return knn_idx, feature, R_matrix1, R_matrix2

    def forward(self, x, normals):
        # batch_size = x.size(0)
        idx, x, R1, R2 = self.get_graph_feature(x, k=self.k, normals=normals)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = x1.transpose(2, 1).contiguous()

        _, x, _, _ = self.get_graph_feature(x1, k=self.k, idx=idx)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        _, x, _, _ = self.get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0].transpose(2, 1).contiguous()

        x = torch.cat((x1, x2, x3), dim=-1)

        return x, R1, R2
