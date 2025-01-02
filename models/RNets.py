import torch
import torch.nn as nn
import torch.nn.functional as F


class Rotation_fix_axis_mirror(nn.Module):
    def __init__(self):
        super(Rotation_fix_axis_mirror, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 256, 1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 1)

        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(16)

    def rotation_matrix_from_axis(self, axis, angle):
        axis = axis.view(-1, 1, 3)
        # 构建旋转矩阵
        skew_symmetric = torch.stack([torch.zeros_like(axis[..., 0]), -axis[..., 2], axis[..., 1],
                                      axis[..., 2], torch.zeros_like(axis[..., 0]), -axis[..., 0],
                                      -axis[..., 1], axis[..., 0], torch.zeros_like(axis[..., 0])], dim=-1).reshape(
            *axis.shape[:-1], 3, 3)

        skew_symmetric.to(angle.device)

        rotation_matrix = torch.eye(3, device=angle.device).unsqueeze(0) + \
                          torch.sin(angle).unsqueeze(-1).unsqueeze(-1) * skew_symmetric + \
                          (1 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1) * torch.matmul(skew_symmetric,
                                                                                            skew_symmetric)

        return rotation_matrix.reshape(-1, 3, 3)

    def mirroring(self, theta, p):
        mirror_vector = torch.cat((torch.cos(theta), torch.cat((torch.sin(theta), torch.zeros(p.size(0), 1).to(p.device)), dim=-1)), dim=-1)
        # mod = torch.sum(mirror_vector ** 2, dim=-1).view(mirror_vector.size(0), -1, 1)
        mirror_vector = mirror_vector.view(-1, 1, 3)
        inner = torch.matmul(mirror_vector, p)

        new_p = p - 2 * torch.matmul(mirror_vector.transpose(2, 1), inner)
        new_p = torch.cat((new_p, p), dim=1)
        return new_p.transpose(2, 1)

    def forward(self, x, direction):
        patch = x
        x = patch.transpose(2, 1).contiguous()
        # batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dp2(x)
        theta = self.fc3(x)
        R_matrix = self.rotation_matrix_from_axis(direction, theta)

        patch = torch.bmm(R_matrix, patch.transpose(2, 1).contiguous())
        y = F.relu(self.bn1(self.conv1(patch)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = torch.max(y, 2, keepdim=True)[0]
        y = y.view(-1, 256)

        y = F.relu(self.bn5(self.fc4(y)))
        y = self.dp1(y)
        y = F.relu(self.bn6(self.fc5(y)))
        y = self.dp2(y)
        theta = self.fc6(y)

        patch = self.mirroring(theta, patch)

        return patch, R_matrix

class similar_net(nn.Module):
    def __init__(self):
        super(similar_net, self).__init__()
        self.bn1 = nn.BatchNorm1d(16)
        self.Linear1 = nn.Sequential(nn.Conv1d(2, 16, kernel_size=1, bias=False),
                                     self.bn1,
                                     nn.ReLU())
        self.dp1 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(16)
        self.Linear2 = nn.Sequential(nn.Conv1d(16, 16, kernel_size=1, bias=False),
                                     self.bn2,
                                     nn.ReLU())
        self.dp2 = nn.Dropout(0.2)
        self.Linear3 = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.dp1(x)
        x = self.Linear2(x)
        x = self.dp2(x)
        x = self.Linear3(x)

        return x


class Refine_net(nn.Module):
    def __init__(self, args):
        super(Refine_net, self).__init__()
        self.bn1 = nn.BatchNorm1d(args.fdims)
        self.Linear1 = nn.Sequential(nn.Conv1d(3 * args.fdims, args.fdims, kernel_size=1, bias=False),
                                     self.bn1,
                                     nn.ReLU())
        self.dp1 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(128)
        self.Linear2 = nn.Sequential(nn.Conv1d(args.fdims, 128, kernel_size=1, bias=False),
                                     self.bn2,
                                     nn.ReLU())
        self.dp2 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(32)
        self.Linear3 = nn.Sequential(nn.Conv1d(128, 32, kernel_size=1, bias=False),
                                     self.bn3,
                                     nn.ReLU())
        self.dp3 = nn.Dropout(0.2)
        self.Linear4 = nn.Conv1d(32, 3, kernel_size=1)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.dp1(x)
        x = self.Linear2(x)
        x = self.dp2(x)
        x = self.Linear3(x)
        x = self.dp3(x)
        x = self.Linear4(x)
        x = x.transpose(2, 1).contiguous()

        return x
