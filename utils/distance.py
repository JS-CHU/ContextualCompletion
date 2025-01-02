import torch
from torch import sqrt


def knn(x, k):
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx.transpose(2, 1) - inner - xx

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def feature_distance(x, y):
    distance = torch.matmul(x, y.transpose(2, 1))
    return distance


def feature_cosine_distance(x, y):
    inner = torch.matmul(x, y.transpose(2, 1))
    x_mod = sqrt(torch.sum(x ** 2, dim=-1)).view(x.size(0), x.size(1), 1)
    y_mod = sqrt(torch.sum(y ** 2, dim=-1)).view(y.size(0), 1, y.size(1))
    xy = torch.matmul(x_mod, y_mod)
    cosine = inner / xy
    return cosine


def coordinate_distance(x, y):
    inner = -2 * torch.matmul(x, y.transpose(2, 1))
    x_2 = torch.sum(x ** 2, dim=2, keepdim=True)
    y_2 = torch.sum(y ** 2, dim=2, keepdim=True)
    distance = x_2 + inner + y_2.transpose(2, 1)
    return distance
