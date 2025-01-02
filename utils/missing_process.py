from utils.distance import coordinate_distance
import torch
import numpy as np

def remove_points_from_cloud(A, B):
    # 将A和B展平为二维数组以便比较
    A_flat = A.view(-1, 3)
    B_flat = B.view(-1, 3)

    # 使用torch.eq()函数找到A中与B相同的点的索引
    mask = ~(torch.eq(A_flat.unsqueeze(1), B_flat.unsqueeze(0)).all(dim=2).any(dim=1))

    # 使用掩码删除A中的点
    A_filtered = A_flat[mask]

    # 将A重塑回原始形状
    A_filtered = A_filtered.view(A.shape[0], -1, 3)

    return A_filtered


def find_missing(pc, partial, device):
    pc = pc.to(device)
    partial = partial.to(device)

    # 计算partial里最小点间距的平均值
    distance_partial = coordinate_distance(partial, partial)
    # print(-(-distance_partial[0][1527]).topk(2, dim=-1)[0])
    distance_partial = -distance_partial
    distance_partial_top2 = distance_partial.topk(2, dim=-1)[0]
    min_distance_partial = distance_partial_top2.min(dim=-1)[0]
    # print(min_distance_partial)
    # print(min_distance_partial[0][1527])
    mean_distance_partial = - min_distance_partial.mean(dim=-1, keepdim=True)
    max_distance_partial = - min_distance_partial.min(dim=-1, keepdim=True)[0]
    # print(mean_distance_partial)
    # print(max_distance_partial)

    door = (mean_distance_partial + max_distance_partial) / 8
    # print(door)

    # 计算pc中的点在partial中的最小点距离
    pp_distance = coordinate_distance(pc, partial)
    min_distance = pp_distance.min(dim=-1)[0]
    # print(min_distance.shape)
    # print(min_indices.shape)

    # 取出min_distance中大于mean_distance_partial的那些
    missing_mask = min_distance > door
    # print(missing_mask.shape)
    # missing_distance = min_distance[missing_mask]
    # print(missing_distance.shape)

    # missing_mask = missing_mask.view(-1)

    return missing_mask.cpu()


def find_missing_v2(pc, partial, device):

    pc.to(device)
    partial.to(device)

    # 计算最小点间距的平均值
    combine = torch.cat((pc, partial), dim=1)
    distance_combine = coordinate_distance(combine, combine)
    distance_combine = -distance_combine
    distance_combine_top2 = distance_combine.topk(2, dim=-1)[0]
    min_distance_combine = distance_combine_top2.min(dim=-1)[0]

    min_distance_combine = - min_distance_combine
    mean_distance_combine = min_distance_combine.mean(dim=-1, keepdim=True)
    max_distance_combine = min_distance_combine.max(dim=-1, keepdim=True)[0]
    print(mean_distance_combine)

    door = (mean_distance_combine + max_distance_combine) / 2
    # door = mean_distance_combine * 1.7

    # 取出min_distance_combine中大于mean_distance_partial的那些
    missing_mask = min_distance_combine > door
    print(missing_mask.shape)
    missing_distance = min_distance_combine[missing_mask]
    print(missing_distance.shape)

    missing_mask = missing_mask.view(-1)
    true_missing = combine.view(combine.shape[1], -1)[missing_mask, :]
    np.savetxt('E:/true_missing.xyz', true_missing)
    np.savetxt('E:/pc.xyz', pc[0])
    np.savetxt('E:/partial.xyz', partial[0])

    return missing_mask