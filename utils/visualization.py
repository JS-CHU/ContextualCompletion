import numpy as np
import open3d as o3d
import random
import time


def visualize_similar_points(p, q, idx):
    # random.seed(time.time())
    window_p = o3d.visualization.Visualizer()
    window_p.create_window(window_name='PointCloud P', height=800, width=800)
    pcd_p = o3d.geometry.PointCloud()
    pcd_p.points = o3d.utility.Vector3dVector(p)
    p_id = random.randint(0, 100)
    # print(p_id)
    p_colors = np.zeros_like(p)
    p_colors[p_id] = [1, 0, 0]
    pcd_p.colors = o3d.utility.Vector3dVector(p_colors)
    window_p.add_geometry(pcd_p)
    window_p.run()

    window_q = o3d.visualization.Visualizer()
    window_q.create_window(window_name='PointCloud Q', height=800, width=800)
    q_idx = idx[p_id]
    pcd_q = o3d.geometry.PointCloud()
    pcd_q.points = o3d.utility.Vector3dVector(q)
    q_colors = np.zeros_like(p)
    for id in q_idx:
        q_colors[id] = [1, 0, 0]

    pcd_q.colors = o3d.utility.Vector3dVector(q_colors)

    window_q.add_geometry(pcd_q)
    window_q.run()


def visualize_heatmap(p, q, weights):
    random.seed(time.time())
    window_p = o3d.visualization.Visualizer()
    window_p.create_window(window_name='PointCloud P', height=800, width=800)
    pcd_p = o3d.geometry.PointCloud()
    pcd_p.points = o3d.utility.Vector3dVector(p)
    p_id = random.randint(0, 2048)
    # print(p_id)
    p_colors = np.zeros_like(p)
    p_colors[p_id] = [1, 0, 0]
    pcd_p.colors = o3d.utility.Vector3dVector(p_colors)
    window_p.add_geometry(pcd_p)
    window_p.run()

    window_q = o3d.visualization.Visualizer()
    window_q.create_window(window_name='PointCloud Q', height=800, width=800)
    weights = weights[p_id]
    weights = weights / np.max(weights)

    pcd_q = o3d.geometry.PointCloud()
    pcd_q.points = o3d.utility.Vector3dVector(q)
    pcd_q.colors = o3d.utility.Vector3dVector(np.stack((weights, np.zeros_like(weights), np.zeros_like(weights)), axis=-1))

    window_q.add_geometry(pcd_q)
    window_q.run()


def visualize(p):
    # random.seed(time.time())
    window_p = o3d.visualization.Visualizer()
    window_p.create_window(window_name='PointCloud P', height=800, width=800)
    pcd_p = o3d.geometry.PointCloud()
    pcd_p.points = o3d.utility.Vector3dVector(p)
    window_p.add_geometry(pcd_p)
    window_p.run()
