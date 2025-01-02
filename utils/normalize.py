import numpy as np
import torch


def normalize_point_cloud_circle(pc, centroid=None, m=None):
    if centroid is not None and m is not None:
        pc = pc - centroid
        pc = pc / m
        return pc, centroid, m
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def normalize_point_cloud_bbox(point_cloud, centroid=None, diagonal_length=None):
    """
    Normalize a 3D point cloud using bounding box diagonal normalization.

    Args:
    - point_cloud (numpy.ndarray): The input point cloud, shape (N, 3), where N is the number of points.

    Returns:
    - normalized_point_cloud (numpy.ndarray): The normalized point cloud.
    """
    if centroid is not None and diagonal_length is not None:
        normalized_point_cloud = (point_cloud - centroid) / diagonal_length
        return normalized_point_cloud, centroid, diagonal_length

    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    # Calculate the bounding box
    centroid = (max_coords + min_coords)/2

    # Calculate the diagonal length of the bounding box
    diagonal_length = np.linalg.norm(max_coords - min_coords)

    # Normalize the point cloud
    normalized_point_cloud = (point_cloud - centroid) / diagonal_length

    return normalized_point_cloud, centroid, diagonal_length


def normalize_point_clouds(pcs, mode):
    shifts = []
    scales = []
    for i in range(pcs.size(0)):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        pc = (pc - shift) / scale
        shifts.append(shift)
        scales.append(scale)
        pcs[i] = pc
    return pcs, shifts, scales

