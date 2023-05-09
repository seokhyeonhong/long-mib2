import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader

from utility.dataset import MotionDataset
from utility.config import Config
from utility import benchmark, utils

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/dataset.json")

    # dataset
    print("Loading dataset...")
    dataset   = MotionDataset(train=False, config=config)
    skeleton  = dataset.skeleton
    v_forward = torch.from_numpy(config.v_forward).to(device)
    
    motion_mean, motion_std = dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # evaluation
    
