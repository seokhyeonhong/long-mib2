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
from model.twostage import ContextTransformer, DetailTransformer

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx_config = Config.load("configs/context_notraj.json")
    det_config = Config.load("configs/detail_notraj.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=False, config=ctx_config)
    dataloader  = DataLoader(dataset, batch_size=ctx_config.batch_size, shuffle=True)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(ctx_config.v_forward).to(device)

    # mean and std
    stat_dset   = MotionDataset(train=True, config=ctx_config)
    motion_mean, motion_std = stat_dset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    # traj_mean, traj_std = stat_dset.traj_statistics()
    # traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    # model
    print("Initializing model...")
    ctx_model = ContextTransformer(len(motion_mean), ctx_config).to(device)
    # ctx_model = ContextTransformer(len(motion_mean), ctx_config, len(traj_mean)).to(device)
    utils.load_model(ctx_model, ctx_config)
    ctx_model.eval()

    det_model = DetailTransformer(len(motion_mean), det_config).to(device)
    # det_model = DetailTransformer(len(motion_mean), det_config, len(traj_mean)).to(device)
    utils.load_model(det_model, det_config)
    det_model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT motion data """
            T = ctx_config.context_frames + 90 + 1
            # T = GT_motion.shape[1]
            GT_motion = GT_motion[:, :T].to(device)
            B, T, D = GT_motion.shape

            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
            # GT_motion[:, -1, :-3] = GT_motion[:, ctx_config.context_frames-1, :-3]
            
            # Optional: interpolate motion
            # GT_motion = utils.get_interpolated_motion(GT_motion, config.context_frames)
            # GT_traj   = utils.get_interpolated_trajectory(GT_traj, ctx_config.context_frames)

            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            """ 2. Forward """
            # forward
            motion = (GT_motion - motion_mean) / motion_std
            # traj   = (GT_traj - traj_mean) / traj_std

            # use traj
            # pred_motion, mask = ctx_model.forward(motion, traj=traj, ratio_constrained=0.0)
            # pred_motion, _    = det_model.forward(pred_motion, mask, traj=traj)

            # no traj
            pred_motion, mask = ctx_model.forward(motion, ratio_constrained=0.0)
            pred_motion, _    = det_model.forward(pred_motion, mask)

            pred_motion = pred_motion * motion_std + motion_mean

            # get motion
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            """ 3. Visualization """
            # animation
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            app = TwoMotionApp(GT_motion, pred_motion, ybot.model(), T, GT_traj.reshape(B*T, -1).cpu().numpy())
            app_manager.run(app)