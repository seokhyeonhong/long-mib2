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
from utility.dataset import MotionDataset
from vis.visapp import KeyframeApp
from model.refinenet import RefineNet

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/refinenet.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = RefineNet(len(motion_mean), len(traj_mean), config, local_attn=False).to(device)
    utils.load_model(model, config)
    model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT data """
            # GT_motion = GT_motion[:, :80]
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)

            GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))
            GT_root_p = GT_global_p[:, :, 0]
            
            """ 2. Forward """
            # normalize - forward - denormalize
            keyframes = model.get_random_keyframes(T)
            motion = model.get_interpolated_motion(GT_motion, keyframes)
            motion = (motion - motion_mean) / motion_std
            traj   = (GT_traj - traj_mean) / traj_std
            pred_motion, pred_contact = model.forward(motion, traj, keyframes)
            pred_motion = pred_motion * motion_std + motion_mean

            # predicted motion
            pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))
            pred_root_p = pred_global_p[:, :, 0]

            """ 3. Animation """
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            total_kfs = copy.deepcopy(keyframes)
            for b in range(1, B):
                for k in keyframes:
                    total_kfs.append(k + b*T)

            app_manager = AppManager()
            app = KeyframeApp(GT_motion, pred_motion, ybot.model(), T, total_kfs)
            app_manager.run(app)