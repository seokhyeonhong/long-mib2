import sys
sys.path.append(".")

import os
import torch
from torch.utils.data import DataLoader

import copy
from tqdm import tqdm

from pymovis.utils import util
from pymovis.motion import Motion, FBX
from pymovis.vis import AppManager
from pymovis.ops import rotation

from utility import utils
from utility.config import Config
from utility.dataset import KeyframeDataset
from vis.visapp import TwoMotionApp
from model.keyframenet import KeyframeNet

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframenet.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = KeyframeDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)
    
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = KeyframeNet(len(motion_mean), len(traj_mean), config).to(device)
    utils.load_model(model, config)
    model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_keyframe in tqdm(dataloader):
            """ 1. GT data """
            GT_keyframe = GT_keyframe.to(device)
            B, T, D = GT_keyframe.shape
            GT_motion, GT_traj, GT_score = torch.split(GT_keyframe, [D-5, 4, 1], dim=-1)

            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-8, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))
            
            """ 2. Forward """
            # normalize - forward - denormalize
            motion = (GT_motion - motion_mean) / motion_std
            traj   = (GT_traj   - traj_mean)   / traj_std
            pred_motion, pred_score = model.forward(motion, traj)
            pred_motion = pred_motion * motion_std + motion_mean

            # predicted motion
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-8, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            """ 3. Animation """
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            app = TwoMotionApp(GT_motion, pred_motion, ybot.model(), T)
            app_manager.run(app)