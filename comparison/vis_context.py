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
from vis.visapp import TwoMotionApp
from model.twostage import ContextTransformer

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config  = Config.load("configs/context_notraj.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=False, config=config)
    dataloader  = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    # mean and std
    stat_dset   = MotionDataset(train=True, config=config)
    motion_mean, motion_std = stat_dset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    # traj_mean, traj_std = stat_dset.traj_statistics()
    # traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    # model
    print("Initializing model...")
    ctx_model = ContextTransformer(len(motion_mean), config).to(device)
    # ctx_model = ContextTransformer(len(motion_mean), config, d_traj=len(traj_mean)).to(device)
    utils.load_model(ctx_model, config)
    ctx_model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT motion data """
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T].to(device)
            B, T, D = GT_motion.shape

            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)

            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            """ 2. Train KF-VAE """
            # forward
            motion = (GT_motion - motion_mean) / motion_std
            # traj   = (GT_traj - traj_mean) / traj_std
            pred_motion, mask = ctx_model.forward(motion, ratio_constrained=0.0)
            # pred_motion, mask = ctx_model.forward(motion, traj, ratio_constrained=0.0)
            pred_motion = pred_motion * motion_std + motion_mean

            # get motion
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            # animation
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            app = TwoMotionApp(GT_motion, pred_motion, ybot.model(), T)
            app_manager.run(app)