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
from model.mrnet import MotionRefineNet

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/dataset.json")
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
    mpvae_cfg = Config.load("configs/mpvae.json")
    mpvae = MotionPredictionVAE(dataset.shape[-1], mpvae_cfg).to(device)
    utils.load_model(mpvae, mpvae_cfg, 150000)
    mpvae.eval()

    mrnet_cfg = Config.load("configs/mrnet.json")
    mrnet = MotionRefineNet(dataset.shape[-1], mrnet_cfg).to(device)
    utils.load_model(mrnet, mrnet_cfg)
    mrnet.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT motion data """
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)

            # batch = (GT_motion - motion_mean) / motion_std

            """ 2. For each batch """
            for b in range(B):
                batch = GT_motion[b:b+1]
                start_frame = 0
                # while start_frame + config.window_length < T:
                while True:
                    ctx_frame = start_frame + config.context_frames - 1
                    end_frame = start_frame + config.window_length

                    """ 2-1. Align MP-VAE input to origin and forward at the context frame """
                    R_diff, root_p_diff = utils.get_align_Rp(batch, ctx_frame, v_forward)
                    batch = utils.align_motion(batch, R_diff, root_p_diff)
                    traj = utils.get_trajectory(batch, v_forward)

                    """ 3. Interpolation """
                    interp_motion = utils.interp_motion(GT_motion[b:b+1], start_frame, end_frame, config.window_length)
                    interp_traj = utils.get_trajectory(interp_motion, v_forward)

                    """ 4. Train MotionRefineNet """
                    # forward
                    batch = (interp_motion - motion_mean) / motion_std
                    pred_motion, pred_contact = mrnet.forward(batch, GT_traj)
                    pred_motion = pred_motion * motion_std + motion_mean

                    # motion features
                    pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
                    pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(1, -1, 3, 3))

                    # animation
                    GT_local_R = GT_local_R[b:b+1, start_frame:end_frame]
                    GT_root_p = GT_root_p[b:b+1, start_frame:end_frame]
                    pred_local_R = pred_local_R.reshape(1, -1, 3, 3)
                    pred_root_p = pred_root_p.reshape(1, -1)

                    GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
                    pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

                    app_manager = AppManager()
                    app = ContextMotionApp(GT_motion, pred_motion, ybot.model(), config.window_length)
                    app_manager.run(app)

                    start_frame += config.window_length










            """ 2. Forward MP-VAE """
            # forward
            pred_motion = mpvae.sample(batch, GT_traj)
            pred_motion = pred_motion * motion_std + motion_mean

            """ 4. Train MotionRefineNet """
            # forward
            batch = (interp_motion - motion_mean) / motion_std
            pred_motion, pred_contact = model.forward(batch, mask, GT_traj)
            pred_motion = pred_motion * motion_std + motion_mean

            # motion features
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

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