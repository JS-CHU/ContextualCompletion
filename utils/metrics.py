import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
from pytorch_structural_losses.nn_distance import nn_distance
import numpy as np

chamfer_dist = chamfer_3DDist()

def CD(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def CD_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def F_score(pred, gt, th=0.01):
    def _get_open3d_ptcloud(tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = o3d.geometry.PointCloud()
        ptcloud.points = o3d.utility.Vector3dVector(tensor)

        return ptcloud

    b = pred.size(0)
    assert pred.size(0) == gt.size(0)
    if b != 1:
        f_score_list = []
        for idx in range(b):
            f_score_list.append(F_score(pred[idx:idx+1], gt[idx:idx+1], th))
        return sum(f_score_list)/len(f_score_list)
    else:
        pred = _get_open3d_ptcloud(pred)
        gt = _get_open3d_ptcloud(gt)

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))
        return 2 * recall * precision / (recall + precision) if recall + precision else 0

def F2_score(pred, gt, th=0.01):
    def _get_open3d_ptcloud(tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = o3d.geometry.PointCloud()
        ptcloud.points = o3d.utility.Vector3dVector(tensor)

        return ptcloud

    b = pred.size(0)
    assert pred.size(0) == gt.size(0)
    if b != 1:
        f_score_list = []
        for idx in range(b):
            f_score_list.append(F2_score(pred[idx:idx+1], gt[idx:idx+1], th))
        return sum(f_score_list)/len(f_score_list)
    else:
        pred = _get_open3d_ptcloud(pred)
        gt = _get_open3d_ptcloud(gt)

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d**2 < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d**2 < th for d in dist1)) / float(len(dist1))
        return 2 * recall * precision / (recall + precision) if recall + precision else 0


def minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, device=None):

    # def iterate_in_chunks(l, n):
    #     '''Yield successive 'n'-sized chunks from iterable 'l'.
    #     Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    #     '''
    #     for i in range(0, len(l), n):
    #         yield l[i:i + n]

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')

    matched_dists = []
    # for i in range(n_ref):
    #     best_in_all_batches = []
    #     ref = torch.from_numpy(ref_pcs[i]).unsqueeze(0).to(device).contiguous()
    #     for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
    #         chunk = torch.from_numpy(sample_chunk).to(device).contiguous()
    #         ref_to_s, s_to_ref = nn_distance(ref, chunk)
    #         all_dist_in_batch = ref_to_s.mean(dim=1) + s_to_ref.mean(dim=1)
    #         best_in_batch = torch.min(all_dist_in_batch).item()
    #         best_in_all_batches.append(best_in_batch)
    #
    #     matched_dists.append(np.min(best_in_all_batches))
    # for i in range(n_ref):
    ref_to_s, s_to_ref = nn_distance(ref_pcs, sample_pcs)
    all_dist_in_batch = ref_to_s.mean(dim=1) + s_to_ref.mean(dim=1)
    mmd = torch.min(all_dist_in_batch).item()

    return mmd, matched_dists


def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """

    :param point_cloud1: (B, N, 3)
    :param point_cloud2: (B, M, 3)
    :return: directed hausdorff distance, A -> B
    """
    point_cloud1 = point_cloud1.transpose(2, 1)
    point_cloud2 = point_cloud2.transpose(2, 1)

    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist


def one_uhd(existing, gen_pcs):
    batch_size = existing.shape[0]
    uhd_list = []
    for i in range(batch_size):
        existing_pc = existing[i]
        gen_pc = gen_pcs[i]
        existing_pc_tensor = existing_pc.repeat((gen_pc.size(0), 1, 1))
        uhd = directed_hausdorff(existing_pc_tensor, gen_pc, reduce_mean=True).item()
        uhd_list.append(uhd)
        # np.savetxt('existing_pc.txt', existing_pc.cpu().numpy())
        # np.savetxt('gen_pc.txt', gen_pc[0].cpu().numpy())
    return np.mean(uhd_list)


def completeness(query_points, ref_points, thres=0.03):

    def nn_distance(query_points, ref_points):
        ref_points_kd_tree = KDTree(ref_points)
        one_distances, one_vertex_ids = ref_points_kd_tree.query(query_points)
        return one_distances
    batch_size = query_points.shape[0]
    percentage_list = []
    for i in range(batch_size):
        query_point = query_points[i].cpu().numpy()
        ref_point = ref_points[i].cpu().numpy()
        a2b_nn_distance =  nn_distance(query_point, ref_point)
        percentage_list.append(np.sum(a2b_nn_distance < thres) / len(a2b_nn_distance))
    error = 1 - sum(percentage_list) / batch_size
    return error


def compute_trimesh_chamfer(gt_points, gen_points, offset=0, scale=1):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: numpy array. trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: numpy array. trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """

    # gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def TMD(gen_pcs):
    mean_dist = 0
    bc = gen_pcs.shape[0]
    l = gen_pcs.shape[1]
    for i in range(bc):
        batch_dist = 0
        # gen_pcs = gen_pcs.cpu().numpy()
        for j in range(l):
            for k in range(j + 1, l, 1):
                pc1 = gen_pcs[i][j].unsqueeze(0)
                pc2 = gen_pcs[i][k].unsqueeze(0)
                chamfer_dist = CD(pc1, pc2)
                batch_dist += chamfer_dist
        mean_batch_dist = batch_dist * 2 / (l - 1)
        mean_dist += mean_batch_dist
    mean_dist = mean_dist / bc
    return mean_dist

