import numpy as np
import random
import torch


def farthest_point_sample_idx(xyz, npoint):
    """
    Input:
        xyz: pointcloud dataset, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    N, C = xyz.shape
    centroids = np.zeros(npoint, dtype=int)
    # print(centroids)
    distance = np.ones(N) * 1e10
    # print(distance)

    farthest = random.randint(0, N-1)
    # print(farthest)
    # 直到采样点达到npoint，否则进行如下迭代：
    for i in range(npoint):
        # 设当前的采样点centroids为当前的最远点farthest
        centroids[i] = farthest
        # print(farthest)
        # print(centroids)
        # 取出该中心点centroid的坐标
        centroid = xyz[farthest, :].reshape(1, 3)
        # print(centroid)
        # 求出所有点到该centroid点的欧式距离，存在dist矩阵中
        dist = np.sum((xyz - centroid) ** 2, -1)
        # print(dist)
        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
        # 随着迭代的继续，distance矩阵中的值会慢慢变小，
        mask = dist < distance # 确保拿到的是距离所有已选中心点最大的距离。比如已经是中心的点，其dist始终保持为	 #0，二在它附近的点，也始终保持与这个中心点的距离
        # print(mask)
        distance[mask] = dist[mask]
        # 从distance矩阵取出最远的点为farthest，继续下一轮迭代
        # print(np.argmax(distance, -1))
        farthest = np.argmax(distance)
    return centroids


def farthest_point_sample_np(point, npoint):
    """
    Input:
        xyz: pointcloud dataset, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N-1)
    for i in range(npoint):
        # print('Sampling point', i)
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    points = point[centroids.astype(np.int32)]
    return points


def farthest_point_sample_tensor(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    sampled_points = torch.zeros(B, npoint, 3).to(device)
    # 初始化一个centroids矩阵，用于存储npoint个采样点的索引位置，大小为B×npoint
    # 其中B为BatchSize的个数
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # distance矩阵(B×N)记录某个batch中所有点到某一个点的距离，初始化的值很大，后面会迭代更新
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个；每个batch都随机有一个初始最远点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices初始化为0~(B-1)的数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 直到采样点达到npoint，否则进行如下迭代：
    for i in range(npoint):
        # 设当前的采样点centroids为当前的最远点farthest
        centroids[:, i] = farthest
        sampled_points[:, i, :] = xyz[batch_indices, farthest, :]
        # 取出该中心点centroid的坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 求出所有点到该centroid点的欧式距离，存在dist矩阵中
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
        # 随着迭代的继续，distance矩阵中的值会慢慢变小，
        # 其相当于记录着某个Batch中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance  # 确保拿到的是距离所有已选中心点最大的距离。比如已经是中心的点，其dist始终保持为	 #0，二在它附近的点，也始终保持与这个中心点的距离
        distance[mask] = dist[mask]
        # 从distance矩阵取出最远的点为farthest，继续下一轮迭代
        farthest = torch.max(distance, -1)[1]
    return sampled_points, centroids

