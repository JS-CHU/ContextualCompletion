# no concat frestore

# from math import exp
import torch
import time
# from torch import sqrt
import torch.nn as nn
import torch.nn.functional as F
from utils.distance import feature_distance, coordinate_distance, feature_cosine_distance
from models.DGCNN import DGCNN_simple, DGCNN_3layers_rotate, DGCNN_3layers
from models.MLP import Refine_net
from utils.fps import farthest_point_sample_tensor
import open3d as o3d
import numpy as np
from models.MLP import similar_net


def get_topk_mu_sigma(x, y, k):
    distance = -coordinate_distance(x, y)
    distance_k = -distance.topk(k=k, dim=-1)[0]
    mean = torch.mean(distance_k, dim=-1).view(distance_k.size(0), distance_k.size(1), -1)
    std = torch.std(distance_k, dim=-1).view(distance_k.size(0), distance_k.size(1), -1)
    return mean, std


def get_feature_similarity(x, y):  # 计算特征相似度（点积）
    distance = feature_distance(x, y)
    # idx = distance.topk(k=k, dim=-1)[1]
    # distance = torch.exp(- distance ** 2)
    return distance


def get_feature_similarity_cosine(x, y):  # 计算特征相似度（余弦）
    distance = feature_cosine_distance(x, y)
    # distance = torch.exp(-distance)
    # idx = distance.topk(k=k, dim=-1)[1]
    # distance = torch.exp(- distance ** 2)
    return distance


def get_coordinate_similarity(x, y):  # 计算坐标相似度（欧氏距离）
    distance = coordinate_distance(x, y)
    # idx = distance.topk(k=k, dim=-1)[1]
    # distance = torch.exp(- distance ** 2)
    return distance


def get_similarity(w, z, k):  # pool k个最相似的 cat(avgpool, maxpool)
    batch_size = z.size(0)
    num_z = z.size(1)
    num_f = w.size(1)
    device = z.device
    idx = w.topk(k=k, dim=-1)[1]
    similar_idx = idx
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_z
    idx = idx + idx_base
    idx = idx.view(-1)
    _, _,  num_dims = z.size()

    feature = z.view(batch_size * num_z, -1)[idx, :]
    feature = feature.view(batch_size, num_f, k, num_dims)
    feature_max = F.adaptive_max_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
    feature_avg = F.adaptive_avg_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
    feature = torch.cat((feature_avg, feature_max), dim=2)

    return feature, similar_idx


def get_weighted_similarity(w, z, k):  # weighted k个最相似的按w做加权 cat(weighted, maxpool)
    batch_size = z.size(0)
    num_z = z.size(1)
    num_f = w.size(1)
    device = z.device
    top_w = w.topk(k=k, dim=-1)
    idx = top_w[1]
    w = top_w[0]
    w = F.softmax(w, dim=2)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_z
    idx = idx + idx_base
    idx = idx.view(-1)
    _, _, num_dims = z.size()

    feature = z.view(batch_size * num_z, -1)[idx, :]
    feature = feature.view(batch_size, num_f, k, num_dims)
    feature_max = F.adaptive_max_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
    w = w.view(batch_size, num_f, k, 1)
    feature = torch.sum(w * feature, 2)

    feature = torch.cat((feature, feature_max), dim=2)

    return feature


def aggregation(f1, f2, p, q, similar_num):
    w1 = get_feature_similarity_cosine(f2, f1)
    w2 = - get_coordinate_similarity(q, p)
    # w = w1 * w2
    # w1 = F.softmax(w1, dim=2)
    # w2 = F.softmax(w2, dim=2)
    # w = w1 + 0.2 * w2
    # idx1 = w1.topk(k=64, dim=-1)[1]
    # idx2 = w2.topk(k=64, dim=-1)[1]
    # idx = w.topk(k=64, dim=-1)[1]
    f, similar_idx = get_similarity(w1, f1, similar_num)
    return f, similar_idx


def aggregation_door(f1, f2, p, q, similar_num):
    w1 = get_feature_similarity_cosine(f2, f1)
    w2 = - get_coordinate_similarity(q, p)
    # w = w1 * w2
    # w1 = F.softmax(w1, dim=2)
    # w2 = F.softmax(w2, dim=2)
    # w = w1 + 0.2 * w2
    # idx1 = w1.topk(k=64, dim=-1)[1]
    # idx2 = w2.topk(k=64, dim=-1)[1]
    # idx = w.topk(k=64, dim=-1)[1]
    f, similar_idx = get_similarity(w1, f1, similar_num)
    return f, similar_idx


def aggregation_cosine_weighted(f1, f2, p, q, similar_num):
    w1 = get_feature_similarity_cosine(f2, f1)
    w2 = get_coordinate_similarity(q, p)
    w1 = torch.exp(w1)
    w2 = torch.exp(-w2)
    # w1 = F.softmax(w1, dim=2)
    # w2 = F.softmax(-w2, dim=2)
    w = w1 * w2

    f = get_weighted_similarity(w, f1, similar_num)
    return f

def aggregation_MLP(f1, f2, p, q, similar_num, similar_net):
    batch_size = p.size(0)
    w1 = get_feature_similarity_cosine(f2, f1).view(batch_size, 1, -1)
    w2 = - get_coordinate_similarity(q, p).view(batch_size, 1, -1)
    # w = w1 * w2
    # w = w1 + w2
    w = torch.cat((w1, w2), dim=1)
    w = similar_net(w).view(batch_size, q.size(1), p.size(1))
    # idx1 = w1.topk(k=64, dim=-1)[1]
    # idx2 = w2.topk(k=64, dim=-1)[1]
    # idx = w.topk(k=64, dim=-1)[1]
    f, similar_idx = get_weighted_similarity(w, f1, similar_num)
    return f, similar_idx


def get_part_query(p, q, fsp, fsq, query_num):
    q_mean, q_std = get_topk_mu_sigma(q, p, query_num)
    q_mean = torch.exp(q_mean)
    q_std = torch.exp(q_std)

    sf = get_feature_similarity_cosine(fsq, fsp)
    sc = -get_coordinate_similarity(q, p)
    sf = torch.max(sf, dim=-1, keepdim=True)[0]
    sc = torch.max(sc, dim=-1, keepdim=True)[0]
    # exp(+) or exp(-) ?
    sf = torch.exp(-sf)
    sc = torch.exp(-sc)

    query = torch.cat((q_mean, q_std, sf, sc), dim=-1)
    query = query.transpose(2, 1).contiguous()

    return query


def get_part_from_score(q, fsq, score, batch_size, num_points, part_num):
    idx = score.topk(part_num, dim=1)[1]
    idx_base = torch.arange(0, batch_size, device=q.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    q_part = q.view(batch_size * num_points, -1)[idx, :]
    q_part = q_part.view(batch_size, part_num, -1)

    fs_qpart = fsq.view(batch_size * num_points, -1)[idx, :]
    fs_qpart = fs_qpart.view(batch_size, part_num, -1)

    return q_part, fs_qpart


def Mirroring(p, mirror_vector):
    mod = torch.sum(mirror_vector ** 2, dim=-1).view(mirror_vector.size(0), -1, 1)
    inner = torch.matmul(mirror_vector, p)

    new_p = p - 2 * torch.matmul(mirror_vector.transpose(2, 1).contiguous(), inner) / mod
    new_p = torch.cat((new_p, p), dim=1)
    return new_p.transpose(2, 1).contiguous()


class RestoreNet(nn.Module):
    def __init__(self, args):
        super(RestoreNet, self).__init__()
        self.similar_num = args.similar_num
        # self.similarity_encoder = DGCNN_simple(args)
        self.feature_encoder = DGCNN_simple(args)
        # self.global_net = PointNet(args, output_channels=256)
        # self.mirror_leaner = Mirror_learner(args)
        # self.score_learner = Score_learner()
        self.refine_net1 = Refine_net(args)
        # self.refine_net2 = Refine_net2(args)
        # self.global_net = PointNet(args, output_channels=256)

    def forward(self, p, q):
        num_points = q.size(1)
        q = torch.cat((p, q), dim=1)
        q = farthest_point_sample_tensor(q, num_points)

        ffp = self.feature_encoder(p) # batch_size * q_num * fdim
        # fsp = self.similarity_encoder(p) # batch_size * p_num * fdim
        ffq = self.feature_encoder(q) # batch_size * q_num * fdim
        # fsq = self.similarity_encoder(q) # batch_size * q_num * fdim
        f1 = aggregation(ffp, ffq, p, q, self.similar_num)
        f1 = torch.cat((ffq, f1, q), dim=-1)
        f1 = f1.transpose(2, 1).contiguous()
        vector = self.refine_net1(f1)
        q_refine = q + vector

        restored = torch.cat((q_refine, p), dim=1)
        restored = farthest_point_sample_tensor(restored, num_points)

        return q_refine, restored


class RestoreNet_rotate_back(nn.Module):
    def __init__(self, args):
        super(RestoreNet_rotate_back, self).__init__()
        self.similar_num = args.similar_num
        self.feature_encoder = DGCNN_3layers_rotate(args)
        self.refine_net = Refine_net(args)

    def compute_normals(self, point_cloud):

        pc = point_cloud.view(-1, 3)
        # 将 PyTorch tensor 转换为 Open3D 点云数据结构
        pc_np = pc.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        # 计算法向量
        radius = 0.1  # 搜索半径
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

        # 将法向量转换为 PyTorch tensor
        normals = torch.tensor(pcd.normals, dtype=torch.float)
        normals = normals.view(point_cloud.shape[0], point_cloud.shape[1], -1)

        return normals.to(point_cloud.device)

    def forward(self, p, q):
        start_Short = time.time()
        batch_size = p.size(0)
        num_points = q.size(1)
        q = torch.cat((p, q), dim=1)
        q, centroids = farthest_point_sample_tensor(q, num_points)
        q_cat = q
        end_Short = time.time()
        execution_Short = end_Short - start_Short
        print("     Short-range Extraction: " + str(execution_Short) + 's')
        p_normals = self.compute_normals(p)
        q_normals = self.compute_normals(q)

        start_Long = time.time()
        ffp, _, _ = self.feature_encoder(p, p_normals) # batch_size * q_num * fdim
        ffq, R1, R2 = self.feature_encoder(q, q_normals) # batch_size * q_num * fdim
        end_Long = time.time()
        execution_Long = end_Long - start_Long
        print("     Long-range Extraction (main geometry processing): " + str(execution_Long) + 's')

        start_refine = time.time()
        f1, similar_idx = aggregation(ffp, ffq, p, q, self.similar_num)
        f1 = torch.cat((ffq, f1), dim=-1)
        f1 = f1.transpose(2, 1).contiguous()
        vector = self.refine_net(f1).view(batch_size, num_points, 3, 1)
        R1 = torch.inverse(R1)
        R2 = torch.inverse(R2)
        R1 = R1.view(batch_size, num_points, 3, 3)
        R2 = R2.view(batch_size, num_points, 3, 3)
        vector = torch.matmul(R1, vector)
        vector = torch.matmul(R2, vector).view(batch_size, num_points, 3)

        mask = centroids >= p.size(1)
        mask = mask.float().unsqueeze(-1)

        q_refine = q + vector * mask
        end_refine = time.time()
        execution_refine = end_refine - start_refine
        print("     Similarity-based Refinement: " + str(execution_refine) + 's')

        # restored = torch.cat((q_refine, p), dim=1)

        return q_cat, q_refine, q_refine, similar_idx


class RestoreNet_rotate_back_similar_MLP(nn.Module):
    def __init__(self, args):
        super(RestoreNet_rotate_back_similar_MLP, self).__init__()
        self.similar_num = args.similar_num
        self.feature_encoder = DGCNN_3layers_rotate(args)
        self.refine_net = Refine_net(args)
        self.similar_net = similar_net()

    def compute_normals(self, point_cloud):
        pc = point_cloud.view(-1, 3)
        # 将 PyTorch tensor 转换为 Open3D 点云数据结构
        pc_np = pc.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        # 计算法向量
        radius = 0.1  # 搜索半径
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

        # 将法向量转换为 PyTorch tensor
        normals = torch.tensor(pcd.normals, dtype=torch.float)
        normals = normals.view(point_cloud.shape[0], point_cloud.shape[1], -1)

        return normals.to(point_cloud.device)

    def get_similarity(self, w, z, k):  # pool k个最相似的 cat(avgpool, maxpool)
        batch_size = z.size(0)
        num_z = z.size(1)
        num_f = w.size(1)
        device = z.device
        idx = w.topk(k=k, dim=-1)[1]
        # idx = w.sort(dim=-1)[1][:, :k]
        similar_idx = idx
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_z
        idx = idx + idx_base
        idx = idx.view(-1)
        _, _, num_dims = z.size()

        feature = z.view(batch_size * num_z, -1)[idx, :]
        feature = feature.view(batch_size, num_f, k, num_dims)
        feature_max = F.adaptive_max_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature_avg = F.adaptive_avg_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature = torch.cat((feature_avg, feature_max), dim=2)

        return feature, similar_idx

    def get_weighted_similarity(self, w, z, k):  # weighted k个最相似的按w做加权 cat(weighted, maxpool)
        batch_size = z.size(0)
        num_z = z.size(1)
        num_f = w.size(1)
        device = z.device
        idx = w.topk(k=k, dim=-1)[1]
        similar_idx = idx
        w = w.gather(2, idx)
        w = F.softmax(w, dim=2)

        # top_w = w.topk(k=k, dim=-1)
        # idx = top_w[1]
        # similar_idx = idx
        # w = top_w[0]

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_z
        idx = idx + idx_base
        idx = idx.view(-1)
        _, _, num_dims = z.size()

        feature = z.view(batch_size * num_z, -1)[idx, :]
        feature = feature.view(batch_size, num_f, k, num_dims)
        feature_max = F.adaptive_max_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        w = w.view(batch_size, num_f, k, 1)
        feature = torch.sum(w * feature, 2)

        feature = torch.cat((feature, feature_max), dim=2)

        return feature, similar_idx

    def aggregation_MLP(self, f1, f2, p, q, similar_num):
        batch_size = p.size(0)
        w1 = get_feature_similarity_cosine(f2, f1).view(batch_size, 1, -1)
        w2 = - get_coordinate_similarity(q, p).view(batch_size, 1, -1)
        # w = w1 * w2
        # w = w1 + w2
        w = torch.cat((w1, w2), dim=1)
        w = self.similar_net(w)
        w = w.view(batch_size, q.size(1), p.size(1))
        w1 = w1.view(batch_size, q.size(1), p.size(1))
        w2 = w2.view(batch_size, q.size(1), p.size(1))
        idx1 = w1.topk(k=64, dim=-1)[1]
        idx2 = w2.topk(k=64, dim=-1)[1]
        idx = w.topk(k=64, dim=-1)[1]
        f, similar_idx = self.get_weighted_similarity(w, f1, similar_num)
        return f, similar_idx, w

    def forward(self, p, q):
        batch_size = p.size(0)
        num_points = q.size(1)
        q = torch.cat((p, q), dim=1)
        q, centroids = farthest_point_sample_tensor(q, num_points)
        q_cat = q

        p_normals = self.compute_normals(p)
        q_normals = self.compute_normals(q)

        ffp, _, _ = self.feature_encoder(p, p_normals) # batch_size * q_num * fdim
        ffq, R1, R2 = self.feature_encoder(q, q_normals) # batch_size * q_num * fdim
        f, similar_idx, w = self.aggregation_MLP(ffp, ffq, p, q, self.similar_num)
        f = torch.cat((ffq, f), dim=-1)
        f = f.transpose(2, 1).contiguous()
        vector = self.refine_net(f).view(batch_size, num_points, 3, 1)
        R1 = torch.inverse(R1)
        R2 = torch.inverse(R2)
        R1 = R1.view(batch_size, num_points, 3, 3)
        R2 = R2.view(batch_size, num_points, 3, 3)
        vector = torch.matmul(R1, vector)
        vector = torch.matmul(R2, vector).view(batch_size, num_points, 3)

        mask = centroids >= p.size(1)
        mask = mask.float().unsqueeze(-1)

        q_refine = q + vector * mask

        # restored = torch.cat((q_refine, p), dim=1)

        return q_cat, q_refine, q_refine, similar_idx, w


class RestoreNet_rotate_back_similar_gate(nn.Module):
    def __init__(self, args):
        super(RestoreNet_rotate_back_similar_gate, self).__init__()
        self.similar_num = args.similar_num
        self.feature_encoder = DGCNN_3layers_rotate(args)
        self.refine_net = Refine_net(args)

    def compute_normals(self, point_cloud):
        pc = point_cloud.view(-1, 3)
        # 将 PyTorch tensor 转换为 Open3D 点云数据结构
        pc_np = pc.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        # 计算法向量
        radius = 0.1  # 搜索半径
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

        # 将法向量转换为 PyTorch tensor
        normals = torch.tensor(np.array(pcd.normals), dtype=torch.float)
        normals = normals.view(point_cloud.shape[0], point_cloud.shape[1], -1)

        return normals.to(point_cloud.device)

    def get_similarity(self, w, z, k):  # pool k个最相似的 cat(avgpool, maxpool)
        batch_size = z.size(0)
        num_z = z.size(1)
        num_f = w.size(1)
        device = z.device
        idx = w.topk(k=k, dim=-1)[1]
        # idx = w.sort(dim=-1)[1][:, :k]
        similar_idx = idx
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_z
        idx = idx + idx_base
        idx = idx.view(-1)
        _, _, num_dims = z.size()

        feature = z.view(batch_size * num_z, -1)[idx, :]
        feature = feature.view(batch_size, num_f, k, num_dims)
        feature_max = F.adaptive_max_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature_avg = F.adaptive_avg_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature = torch.cat((feature_avg, feature_max), dim=2)

        return feature, similar_idx

    def aggregation(self, f1, f2, p, q, similar_num):
        batch_size = p.size(0)
        w1 = get_feature_similarity_cosine(f2, f1)
        # w2 = torch.exp(get_coordinate_similarity(q, p))
        w2 = torch.exp(-get_coordinate_similarity(q, p))
        coor_num = int(similar_num / 4)
        # _, w2_indices = w2.sort(dim=-1)
        # w2_indices = w2_indices[:, :, :coor_num]
        _, w2_indices = w2.topk(k=coor_num, dim=-1)
        zeros = torch.zeros_like(w2)
        # w = w2.gather(2, w2_indices)
        zeros.scatter_(2, w2_indices, w2.gather(2, w2_indices))
        # w = torch.exp(w1) + (-zeros)
        w = torch.exp(w1) + zeros
        # w = w1 * w2
        # w = w1 + w2
        # idx1 = w1.topk(k=64, dim=-1)[1]
        # idx2 = zeros.topk(k=64, dim=-1)[1]
        # idx = w.topk(k=64, dim=-1)[1]
        f, similar_idx = self.get_similarity(w, f1, similar_num)
        return f, similar_idx, w

    def forward(self, p, q):
        batch_size = p.size(0)
        num_points = q.size(1)
        p, _ = farthest_point_sample_tensor(p, num_points)
        # q = torch.cat((p, q), dim=1)
        # q, centroids = farthest_point_sample_tensor(q, num_points)
        q_cat = q

        p_normals = self.compute_normals(p)
        q_normals = self.compute_normals(q)

        ffp, _, _ = self.feature_encoder(p, p_normals) # batch_size * q_num * fdim
        ffq, R1, R2 = self.feature_encoder(q, q_normals) # batch_size * q_num * fdim
        f, similar_idx, w = self.aggregation(ffp, ffq, p, q, self.similar_num)
        f = torch.cat((ffq, f), dim=-1)
        f = f.transpose(2, 1).contiguous()
        vector = self.refine_net(f).view(batch_size, num_points, 3, 1)
        R1 = torch.inverse(R1)
        R2 = torch.inverse(R2)
        R1 = R1.view(batch_size, num_points, 3, 3)
        R2 = R2.view(batch_size, num_points, 3, 3)
        vector = torch.matmul(R2, vector)
        vector = torch.matmul(R1, vector).view(batch_size, num_points, 3)

        # mask = centroids >= p.size(1)
        # mask = mask.float().unsqueeze(-1)

        q_refine = q + vector# * mask

        # restored = torch.cat((q_refine, p), dim=1)

        return q_cat, q_refine, q_refine, similar_idx, w

class RestoreNet_rotate_back_similar_gate_2(nn.Module):
    def __init__(self, args):
        super(RestoreNet_rotate_back_similar_gate_2, self).__init__()
        self.similar_num = args.similar_num
        self.feature_encoder = DGCNN_3layers_rotate(args)
        self.refine_net = Refine_net(args)

    def compute_normals(self, point_cloud):
        pc = point_cloud.view(-1, 3)
        # 将 PyTorch tensor 转换为 Open3D 点云数据结构
        pc_np = pc.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        # 计算法向量
        radius = 0.1  # 搜索半径
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

        # 将法向量转换为 PyTorch tensor
        normals = torch.tensor(np.array(pcd.normals), dtype=torch.float)
        normals = normals.view(point_cloud.shape[0], point_cloud.shape[1], -1)

        return normals.to(point_cloud.device)

    def get_similarity(self, w, z, k):  # pool k个最相似的 cat(avgpool, maxpool)
        batch_size = z.size(0)
        num_z = z.size(1)
        num_f = w.size(1)
        device = z.device
        idx = w.topk(k=k, dim=-1)[1]
        # idx = w.sort(dim=-1)[1][:, :k]
        similar_idx = idx
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_z
        idx = idx + idx_base
        idx = idx.view(-1)
        _, _, num_dims = z.size()

        feature = z.view(batch_size * num_z, -1)[idx, :]
        feature = feature.view(batch_size, num_f, k, num_dims)
        feature_max = F.adaptive_max_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature_avg = F.adaptive_avg_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature = torch.cat((feature_avg, feature_max), dim=2)

        return feature, similar_idx

    def aggregation(self, f1, f2, p, q, similar_num):
        batch_size = p.size(0)
        w1 = get_feature_similarity_cosine(f2, f1)
        # w2 = torch.exp(get_coordinate_similarity(q, p))
        w2 = torch.exp(-get_coordinate_similarity(q, p))
        coor_num = int(similar_num / 4)
        # _, w2_indices = w2.sort(dim=-1)
        # w2_indices = w2_indices[:, :, :coor_num]
        _, w2_indices = w2.topk(k=coor_num, dim=-1)
        zeros = torch.zeros_like(w2)
        # w = w2.gather(2, w2_indices)
        zeros.scatter_(2, w2_indices, w2.gather(2, w2_indices))
        # w = torch.exp(w1) + (-zeros)
        w = torch.exp(w1) + zeros
        # w = w1 * w2
        # w = w1 + w2
        # idx1 = w1.topk(k=64, dim=-1)[1]
        # idx2 = zeros.topk(k=64, dim=-1)[1]
        # idx = w.topk(k=64, dim=-1)[1]
        f, similar_idx = self.get_similarity(w, f1, similar_num)
        return f, similar_idx, w

    def forward(self, p, q):
        batch_size = p.size(0)
        num_points = q.size(1)
        p_mirror = p.clone()
        p_mirror[:, :, 2] = -p_mirror[:, :, 2]
        # p = torch.cat((p, p_mirror), dim=1)
        # np.savetxt('/Workspace/private/code/completion/logs_results(new)/results_restore/MVP/p.txt', p[0].cpu().numpy())
        # q1 = torch.cat((p, q), dim=1)
        # q1, centroids1 = farthest_point_sample_tensor(q1, num_points)
        # np.savetxt('/Workspace/private/code/completion/logs_results(new)/results_restore/MVP/q1.txt', q1[0].cpu().numpy())
        q = torch.cat((p, p_mirror, q), dim=1)
        q, centroids = farthest_point_sample_tensor(q, num_points)
        q_cat = q
        # np.savetxt('/Workspace/private/code/completion/logs_results(new)/results_restore/MVP/q.txt', q[0].cpu().numpy())
        p_normals = self.compute_normals(p)
        q_normals = self.compute_normals(q)

        ffp, _, _ = self.feature_encoder(p, p_normals) # batch_size * q_num * fdim
        ffq, R1, R2 = self.feature_encoder(q, q_normals) # batch_size * q_num * fdim
        f, similar_idx, w = self.aggregation(ffp, ffq, p, q, self.similar_num)
        f = torch.cat((ffq, f), dim=-1)
        f = f.transpose(2, 1).contiguous()
        vector = self.refine_net(f).view(batch_size, num_points, 3, 1)
        R1 = torch.inverse(R1)
        R2 = torch.inverse(R2)
        R1 = R1.view(batch_size, num_points, 3, 3)
        R2 = R2.view(batch_size, num_points, 3, 3)
        vector = torch.matmul(R2, vector)
        vector = torch.matmul(R1, vector).view(batch_size, num_points, 3)

        mask = centroids >= p.size(1)
        mask = mask.float().unsqueeze(-1)

        q_refine = q + vector * mask

        # restored = torch.cat((q_refine, p), dim=1)

        return q_cat, q_refine, q_refine, similar_idx, w


class RestoreNet_no_rotate(nn.Module):
    def __init__(self, args):
        super(RestoreNet_no_rotate, self).__init__()
        self.similar_num = args.similar_num
        self.feature_encoder = DGCNN_3layers(args)
        self.refine_net = Refine_net(args)

    def compute_normals(self, point_cloud):
        pc = point_cloud.view(-1, 3)
        # 将 PyTorch tensor 转换为 Open3D 点云数据结构
        pc_np = pc.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        # 计算法向量
        radius = 0.1  # 搜索半径
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

        # 将法向量转换为 PyTorch tensor
        normals = torch.tensor(pcd.normals, dtype=torch.float)
        normals = normals.view(point_cloud.shape[0], point_cloud.shape[1], -1)

        return normals.to(point_cloud.device)

    def get_similarity(self, w, z, k):  # pool k个最相似的 cat(avgpool, maxpool)
        batch_size = z.size(0)
        num_z = z.size(1)
        num_f = w.size(1)
        device = z.device
        idx = w.topk(k=k, dim=-1)[1]
        # idx = w.sort(dim=-1)[1][:, :k]
        similar_idx = idx
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_z
        idx = idx + idx_base
        idx = idx.view(-1)
        _, _, num_dims = z.size()

        feature = z.view(batch_size * num_z, -1)[idx, :]
        feature = feature.view(batch_size, num_f, k, num_dims)
        feature_max = F.adaptive_max_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature_avg = F.adaptive_avg_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature = torch.cat((feature_avg, feature_max), dim=2)

        return feature, similar_idx

    def aggregation(self, f1, f2, p, q, similar_num):
        # batch_size = p.size(0)
        w1 = get_feature_similarity_cosine(f2, f1)
        # w2 = torch.exp(get_coordinate_similarity(q, p))
        w2 = torch.exp(-get_coordinate_similarity(q, p))
        coor_num = int(similar_num / 4)
        # _, w2_indices = w2.sort(dim=-1)
        # w2_indices = w2_indices[:, :, :coor_num]
        _, w2_indices = w2.topk(k=coor_num, dim=-1)
        zeros = torch.zeros_like(w2)
        # w = w2.gather(2, w2_indices)
        zeros.scatter_(2, w2_indices, w2.gather(2, w2_indices))
        # w = torch.exp(w1) + (-zeros)
        w = torch.exp(w1) + zeros
        # w = w1 * w2
        # w = w1 + w2
        # idx1 = w1.topk(k=64, dim=-1)[1]
        # idx2 = zeros.topk(k=64, dim=-1)[1]
        # idx = w.topk(k=64, dim=-1)[1]
        f, similar_idx = self.get_similarity(w, f1, similar_num)
        return f, similar_idx, w

    def forward(self, p, q):
        # batch_size = p.size(0)
        num_points = q.size(1)
        q = torch.cat((p, q), dim=1)
        q, centroids = farthest_point_sample_tensor(q, num_points)
        q_cat = q

        # p_normals = self.compute_normals(p)
        # q_normals = self.compute_normals(q)

        ffp = self.feature_encoder(p)  # batch_size * q_num * fdim
        # fsp = self.similarity_encoder(p) # batch_size * p_num * fdim
        ffq = self.feature_encoder(q)  # batch_size * q_num * fdim
        # fsq = self.similarity_encoder(q) # batch_size * q_num * fdim
        f, similar_idx, w = self.aggregation(ffp, ffq, p, q, self.similar_num)
        f = torch.cat((ffq, f), dim=-1)
        f = f.transpose(2, 1).contiguous()
        vector = self.refine_net(f)

        mask = centroids >= p.size(1)
        mask = mask.float().unsqueeze(-1)

        q_refine = q + vector * mask

        return q_cat, q_refine, q_refine, similar_idx, w


class RestoreNet_epn(nn.Module):
    def __init__(self, args):
        super(RestoreNet_epn, self).__init__()
        self.similar_num = args.similar_num
        self.feature_encoder = DGCNN_3layers_rotate(args)
        self.refine_net = Refine_net(args)

    def compute_normals(self, point_cloud):
        pc = point_cloud.view(-1, 3)
        # 将 PyTorch tensor 转换为 Open3D 点云数据结构
        pc_np = pc.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        # 计算法向量
        radius = 0.1  # 搜索半径
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

        # 将法向量转换为 PyTorch tensor
        normals = torch.tensor(np.array(pcd.normals), dtype=torch.float)
        normals = normals.view(point_cloud.shape[0], point_cloud.shape[1], -1)

        return normals.to(point_cloud.device)

    def get_similarity(self, w, z, k):  # pool k个最相似的 cat(avgpool, maxpool)
        batch_size = z.size(0)
        num_z = z.size(1)
        num_f = w.size(1)
        device = z.device
        idx = w.topk(k=k, dim=-1)[1]
        # idx = w.sort(dim=-1)[1][:, :k]
        similar_idx = idx
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_z
        idx = idx + idx_base
        idx = idx.view(-1)
        _, _, num_dims = z.size()

        feature = z.view(batch_size * num_z, -1)[idx, :]
        feature = feature.view(batch_size, num_f, k, num_dims)
        feature_max = F.adaptive_max_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature_avg = F.adaptive_avg_pool2d(feature, (1, num_dims)).view(batch_size, num_f, -1)
        feature = torch.cat((feature_avg, feature_max), dim=2)

        return feature, similar_idx

    def aggregation(self, f1, f2, p, q, similar_num):
        batch_size = p.size(0)
        w1 = get_feature_similarity_cosine(f2, f1)
        # w2 = torch.exp(get_coordinate_similarity(q, p))
        w2 = torch.exp(-get_coordinate_similarity(q, p))
        coor_num = int(similar_num / 4)
        # _, w2_indices = w2.sort(dim=-1)
        # w2_indices = w2_indices[:, :, :coor_num]
        _, w2_indices = w2.topk(k=coor_num, dim=-1)
        zeros = torch.zeros_like(w2)
        # w = w2.gather(2, w2_indices)
        zeros.scatter_(2, w2_indices, w2.gather(2, w2_indices))
        # w = torch.exp(w1) + (-zeros)
        w = torch.exp(w1) + zeros
        # w = w1 * w2
        # w = w1 + w2
        # idx1 = w1.topk(k=64, dim=-1)[1]
        # idx2 = zeros.topk(k=64, dim=-1)[1]
        # idx = w.topk(k=64, dim=-1)[1]
        f, similar_idx = self.get_similarity(w, f1, similar_num)
        return f, similar_idx, w

    def forward(self, p, q):
        batch_size = p.size(0)
        num_points = q.size(1)
        q_cat = q

        p_normals = self.compute_normals(p)
        q_normals = self.compute_normals(q)

        ffp, _, _ = self.feature_encoder(p, p_normals) # batch_size * q_num * fdim
        ffq, R1, R2 = self.feature_encoder(q, q_normals) # batch_size * q_num * fdim
        f, similar_idx, w = self.aggregation(ffp, ffq, p, q, self.similar_num)
        f = torch.cat((ffq, f), dim=-1)
        f = f.transpose(2, 1).contiguous()
        vector = self.refine_net(f).view(batch_size, num_points, 3, 1)
        R1 = torch.inverse(R1)
        R2 = torch.inverse(R2)
        R1 = R1.view(batch_size, num_points, 3, 3)
        R2 = R2.view(batch_size, num_points, 3, 3)
        vector = torch.matmul(R2, vector)
        vector = torch.matmul(R1, vector).view(batch_size, num_points, 3)

        q_refine = q + vector

        # restored = torch.cat((q_refine, p), dim=1)

        return q_cat, q_refine, q_refine, similar_idx, w