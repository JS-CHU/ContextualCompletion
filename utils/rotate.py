import torch
import numpy as np
import random
import torch.nn.functional as F


def rotate_point_cloud(point_cloud, angles):
    """
    Rotate point cloud around x, y, z axes.
    Args:
        point_cloud: Tensor of shape (batch_size, num_points, 3)
        angles: Tuple of three angles for rotation around x, y, z axes
    Returns:
        Rotated point cloud tensor
    """
    # Convert angles to radians
    angles = [angle * (3.141592653589793 / 180.0) for angle in angles]


    # Rotation matrices
    Rx = torch.tensor([[1, 0, 0],
                      [0, np.cos(angles[0]), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])

    Ry = torch.tensor([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])

    Rz = torch.tensor([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])

    # Rotate point cloud
    rotated_point_cloud = torch.matmul(point_cloud, Rx)
    rotated_point_cloud = torch.matmul(rotated_point_cloud, Ry)
    rotated_point_cloud = torch.matmul(rotated_point_cloud, Rz)

    return rotated_point_cloud

def random_rotation_tensor(point_cloud_tensor, rate):
    """
    Randomly rotate a subset of point clouds in a tensor.
    Args:
        point_cloud_tensor: Tensor of shape (batch_size, num_points, 3)
        rate: Proportion of point clouds to rotate
    Returns:
        New tensor with rotated point clouds
    """
    batch_size, num_points, _ = point_cloud_tensor.shape

    # Determine the number of point clouds to rotate
    num_to_rotate = int(rate * batch_size)

    # Randomly choose indices of point clouds to rotate
    indices_to_rotate = random.sample(range(batch_size), num_to_rotate)
    rotate_input =  point_cloud_tensor[indices_to_rotate]

    # Random angles for rotation
    angles = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
    print(angles)

    random_rotated = rotate_point_cloud(rotate_input, angles)
    rotated_point_clouds = torch.cat((point_cloud_tensor, random_rotated), dim=0)

    return rotated_point_clouds


def rotate_pc(pc):
    R = torch.FloatTensor([[0.00, 0.00, -1.00], [0.00, 1.00, 0.00], [1.00, 0.00, 0.00]])
    pc_rotate = torch.matmul(pc, R)
    return pc_rotate.type_as(pc)


def rotate_back_pc(pc):
    R = torch.FloatTensor([[0.00, 0.00, 1.00], [0.00, 1.00, 0.00], [-1.00, 0.00, 0.00]])
    pc_rotate = torch.matmul(pc, R)
    return pc_rotate.type_as(pc)


def get_rotation_matrix(A, B):
    # 单位化向量
    A_unit = F.normalize(A, p=2, dim=-1)
    B_unit = F.normalize(B.unsqueeze(1), p=2, dim=-1)

    # 计算旋转轴
    axis = torch.cross(A_unit, B_unit, dim=-1)

    # 计算旋转角度
    angle = torch.acos(torch.sum(A_unit * B_unit, dim=-1))

    # 构建旋转矩阵
    skew_symmetric = torch.stack([torch.zeros_like(axis[..., 0]), -axis[..., 2], axis[..., 1],
                                  axis[..., 2], torch.zeros_like(axis[..., 0]), -axis[..., 0],
                                  -axis[..., 1], axis[..., 0], torch.zeros_like(axis[..., 0])], dim=-1).reshape(
        *axis.shape[:-1], 3, 3)

    rotation_matrix = torch.eye(3, device=A.device).unsqueeze(0) + \
                      torch.sin(angle).unsqueeze(-1).unsqueeze(-1) * skew_symmetric + \
                      (1 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1) * torch.matmul(skew_symmetric, skew_symmetric)

    return rotation_matrix.reshape(-1, 3, 3)



