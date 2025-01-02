import os
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
from utils.missing_process import find_missing


class vipc_trainset(Dataset):

    def __init__(self, root, cls_list, missing_mode=True):
        self.missing_mode = missing_mode
        self.name = []
        self.file = []
        self.part = []
        self.gt = []
        if missing_mode:
            self.missing_mask = []
        for cls in cls_list:
            gt_dir = os.path.join(root, 'ShapeNetViPC-GT')
            part_dir = os.path.join(root, 'ShapeNetViPC-Partial')
            with open(os.path.join(root, 'train_list.txt')) as f:
                train_files = f.readlines()
            train_files = [x.strip() for x in train_files]
            num = 0
            for path in train_files:
                if cls in path:
                    num += 1
                    part_path = os.path.join(part_dir, path + '_2048.dat')
                    gt_path = os.path.join(gt_dir, path + '_2048.dat')
                    if not os.path.exists(part_path) or not os.path.exists(gt_path):
                        continue
                    part_pc = self.load_dat(part_path)
                    gt_pc = self.load_dat(gt_path)

                    # normalize partial point cloud and GT to the same scale
                    gt_mean = gt_pc.mean(axis=0)
                    gt_pc = gt_pc - gt_mean
                    gt_L_max = np.max(np.sqrt(np.sum(abs(gt_pc ** 2), axis=-1)))
                    gt_pc = gt_pc / gt_L_max

                    part_pc = part_pc - gt_mean
                    part_pc = part_pc / gt_L_max

                    gt_pc = torch.FloatTensor(gt_pc).view(1, gt_pc.shape[0], gt_pc.shape[1])
                    part_pc = torch.FloatTensor(part_pc).view(1, part_pc.shape[0], part_pc.shape[1])
                    self.part.append(part_pc[0])
                    self.gt.append(gt_pc[0])
                    if missing_mode:
                        missing_mask = find_missing(gt_pc, part_pc, 'cuda')
                        self.missing_mask.append(missing_mask[0])
                    self.name.append(cls)
                    self.file.append(path)

    def load_dat(self, path):
        with open(path, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        return pc

    def save_dat(self, path, pc):
        with open(path, 'wb') as f:
            # print('Saving ', path)
            pickle.dump(pc, f)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        if self.missing_mode:
            data = {
                'name': self.name[idx],
                'file': self.file[idx],
                'part': self.part[idx],
                'gt': self.gt[idx],
                'missing_mask': self.missing_mask[idx]
            }
        else:
            data = {
                'name': self.name[idx],
                'file': self.file[idx],
                'part': self.part[idx],
                'gt': self.gt[idx],
            }
        return data


class vipc_valset(Dataset):

    def __init__(self, root, cls_list, missing_mode=True):
        self.missing_mode = missing_mode
        self.name = []
        self.file = []
        self.part = []
        self.gt = []
        self.part_shifts = []
        self.part_scales = []
        self.gt_shifts = []
        self.gt_scales = []
        if missing_mode:
            self.missing_mask = []
        for cls in cls_list:
            gt_dir = os.path.join(root, 'ShapeNetViPC-GT')
            part_dir = os.path.join(root, 'ShapeNetViPC-Partial')
            with open(os.path.join(root, 'test_list.txt')) as f:
                train_files = f.readlines()
            train_files = [x.strip() for x in train_files]
            num = 0
            for path in train_files:
                if cls in path:
                    num += 1
                    part_path = os.path.join(part_dir, path + '_2048.dat')
                    gt_path = os.path.join(gt_dir, path + '_2048.dat')
                    if not os.path.exists(part_path) or not os.path.exists(gt_path):
                        continue
                    part_pc = self.load_dat(part_path)
                    gt_pc = self.load_dat(gt_path)

                    # normalize partial point cloud and GT to the same scale
                    gt_mean = gt_pc.mean(axis=0)
                    gt_L_max = np.max(np.sqrt(np.sum(abs(gt_pc ** 2), axis=-1)))
                    gt_pc = gt_pc - gt_mean
                    gt_pc = gt_pc / gt_L_max

                    part_pc = part_pc - gt_mean
                    part_pc = part_pc / gt_L_max

                    gt_pc = torch.FloatTensor(gt_pc).view(1, gt_pc.shape[0], gt_pc.shape[1])
                    part_pc = torch.FloatTensor(part_pc).view(1, part_pc.shape[0], part_pc.shape[1])

                    self.part.append(part_pc[0])
                    self.gt.append(gt_pc[0])
                    if missing_mode:
                        missing_mask = find_missing(gt_pc, part_pc, 'cuda')
                        self.missing_mask.append(missing_mask[0])
                    self.name.append(cls)
                    self.file.append(path)

    def load_dat(self, path):
        with open(path, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        return pc

    def save_dat(self, path, pc):
        with open(path, 'wb') as f:
            pickle.dump(pc, f)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        if self.missing_mode:
            data = {
                'name': self.name[idx],
                'file': self.file[idx],
                'part': self.part[idx],
                'gt': self.gt[idx],
                'missing_mask': self.missing_mask[idx]
            }
        else:
            data = {
                'name': self.name[idx],
                'file': self.file[idx],
                'part': self.part[idx],
                'gt': self.gt[idx],
            }
        return data
