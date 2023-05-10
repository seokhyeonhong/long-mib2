import os
import pickle
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utility.utils import get_motion

class MotionDataset(Dataset):
    """
    Motion dataset for training and testing
    Features:
        - motion features (number of joints * 6 + 3) for each frame, 6D orientations and a 3D translation vector
        - trajectory features (4) for each frame, a 2D base xz position and a 2D forward vector
    """
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        with open(os.path.join(self.config.dataset_dir, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)

        self.features = torch.from_numpy(np.load(config.trainset_npy if train else config.testset_npy))
        self.shape = self.features.shape
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

    def motion_statistics(self, dim=(0, 1)):
        print(f"Calculating MotionDataset motion_mean and motion_std, dim={dim}...")

        # calculate statistics from training set
        trainset = MotionDataset(True, self.config)

        # mean and std
        X = torch.stack([trainset[i] for i in range(len(trainset))], dim=0)
        X = X[..., :-4]
        mean = torch.mean(X, dim=dim)
        std = torch.std(X, dim=dim) + 1e-8

        return mean, std

    def traj_statistics(self, dim=(0, 1)):
        print(f"Calculating MotionDataset traj_mean and traj_std, dim={dim}...")

        # calculate statistics from training set
        trainset = MotionDataset(True, self.config)

        # mean and std
        X = torch.stack([trainset[i] for i in range(len(trainset))], dim=0)
        X = X[..., -4:]
        mean = torch.mean(X, dim=dim)
        std = torch.std(X, dim=dim) + 1e-8

        return mean, std
    
    def test_statistics(self):
        print(f"Calculating MotionDataset test_mean and test_std...")

        # training set
        trainset = MotionDataset(True, self.config)
        dataloader = DataLoader(trainset, batch_size=self.config.batch_size, shuffle=False)

        # global positions
        global_ps = []
        for batch in dataloader:
            batch = batch[..., :-4]
            _, global_p = get_motion(batch, self.skeleton)
            global_ps.append(global_p)
        global_ps = torch.cat(global_ps, dim=0)
        global_ps = global_ps.reshape(global_ps.shape[0], global_ps.shape[1], -1)

        # mean and std
        mean = torch.mean(global_ps, dim=(0, 1))
        std = torch.std(global_ps, dim=(0, 1)) + 1e-8

        return mean, std

class KeyframeDataset(Dataset):
    """
    Motion dataset for training and testing
    Features:
        - motion features (number of joints * 6 + 3) for each frame, 6D orientations and a 3D translation vector
        - keyframe probability (1) for each frame
        - trajectory features (5) for each frame, xz and forward vector
    """
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        with open(os.path.join(self.config.dataset_dir, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)

        self.features = torch.from_numpy(np.load(config.keyframe_trainset_npy if train else config.keyframe_testset_npy))
        self.shape = self.features.shape
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

    def motion_statistics(self, dim=(0, 1)):
        print(f"Calculating KeyframeDataset motion_mean and motion_std, dim={dim}...")

        # calculate statistics from training set
        trainset = KeyframeDataset(True, self.config)

        # mean and std
        X = torch.stack([trainset[i] for i in range(len(trainset))], dim=0)
        X = X[..., :-5] # 4 for traj, 1 for score
        mean = torch.mean(X, dim=dim)
        std = torch.std(X, dim=dim) + 1e-8

        return mean, std

    def traj_statistics(self, dim=(0, 1)):
        print(f"Calculating KeyframeDataset traj_mean and traj_std, dim={dim}...")

        # calculate statistics from training set
        trainset = KeyframeDataset(True, self.config)

        # mean and std
        X = torch.stack([trainset[i] for i in range(len(trainset))], dim=0)
        X = X[..., -5:-1]
        mean = torch.mean(X, dim=dim)
        std = torch.std(X, dim=dim) + 1e-8

        return mean, std