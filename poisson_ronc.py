import numpy as np
import torch
import open3d as o3d

def poisson_reconstruction(points, depth, device):
    points_np = points.numpy()
    normals = compute_normals(points_np)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh

def compute_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    return np.asarray(pcd.normals)

# 示例用法
if __name__ == "__main__":
    points = np.loadtxt('')
    points = torch.from_numpy(points)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = poisson_reconstruction(points, depth=10, device=device)
    o3d.visualization.draw_geometries([mesh])
