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
from vis.visapp import ContextMotionApp
from model.mpvae import MotionPredictionVAE

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/mpvae.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = MotionPredictionVAE(dataset.shape[-1], config).to(device)
    utils.load_model(model, config, 150000)
    model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT motion data """
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_local_R6, GT_global_p, GT_traj = utils.get_motion_and_trajectory(GT_motion, skeleton, v_forward)
            GT_traj = utils.get_interpolated_trajectory(GT_traj, config.context_frames)

            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))
            GT_root_p  = GT_global_p[:, :, 0, :]

            """ 2. Train KF-VAE """
            # forward
            batch = (GT_motion - motion_mean) / motion_std
            pred_motion = model.sample(batch, GT_traj)
            pred_motion = pred_motion * motion_std + motion_mean

            # get motion
            pred_local_R6, pred_global_p, pred_traj  = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))
            pred_root_p  = pred_global_p[:, :, 0, :]

            # animation
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            app = ContextMotionApp(GT_motion, pred_motion, ybot.model(), T)
            app_manager.run(app)