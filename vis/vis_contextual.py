import sys
sys.path.append(".")

import os
import random
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
from model.contextual import ContextualVAE

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/contextual.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=True, config=config)
    val_dataset = MotionDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = ContextualVAE(len(motion_mean), len(traj_mean), config).to(device)
    utils.load_model(model, config)
    model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            # GT motion data
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # forward
            start_frame, end_frmae = 0, 0
            mask = torch.ones(B, T, 1).to(device)
            mask[:, config.context_frames:-1, :] = 0
            output_motion = GT_motion * mask

            while start_frame + (config.context_frames-1) < T - 1:
                if start_frame + (config.context_frames-1) + config.max_transition + 1 >= T-1:
                    end_frame = T-1
                    constrained = True
                else:
                    end_frame = start_frame + (config.context_frames-1) + random.randint(config.min_transition, config.max_transition)
                    constrained = False
                
                # align
                R_diff, p_diff = utils.get_align_Rp(output_motion, start_frame + (config.context_frames-1), v_forward)
                aligned_motion = utils.align_motion(output_motion, R_diff, p_diff)
                aligned_traj   = utils.align_traj(GT_traj, R_diff, p_diff)

                if constrained:
                    aligned_motion = aligned_motion[:, start_frame:]
                    aligned_traj   = aligned_traj[:, start_frame:]
                else:
                    aligned_motion = torch.cat([aligned_motion[:, start_frame:end_frame+1], aligned_motion[:, -1:]], dim=1)
                    aligned_traj   = torch.cat([aligned_traj[:, start_frame:end_frame+1], aligned_traj[:, -1:]], dim=1)

                # forward
                motion = (aligned_motion - motion_mean) / motion_std
                traj   = (aligned_traj - traj_mean) / traj_std
                pred_motion = model.sample(motion, traj, (T-1) - start_frame, constrained=constrained)
                print((T-1) - start_frame)
                pred_motion = pred_motion * motion_std + motion_mean

                # re-align
                pred_motion = utils.restore_motion(pred_motion, R_diff, p_diff)

                # update
                if constrained:
                    output_motion[:, start_frame:end_frame+1, :] = pred_motion
                else:
                    output_motion[:, start_frame:end_frame+1, :] = pred_motion[:, :-1]

                # next
                if constrained:
                    break
                else:
                    start_frame = end_frame - (config.context_frames-1)

            # prediction
            pred_local_R6, pred_root_p = torch.split(output_motion, [D-7, 3], dim=-1)
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