import torch
from cuda.emd.emd_module import emdFunction
import open3d as o3d
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()


def CD(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def CD_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return d1 + d2


def EMD_loss(preds, gts, eps=0.005, iters=50):
    loss, _ = emdFunction.apply(preds, gts, eps, iters)
    return torch.sum(loss)

