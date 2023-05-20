import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pymovis.ops import motionops, rotation, mathops

from utility.dataset import MotionDataset
from utility.config import Config
from utility import benchmark, utils

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/dataset.json")
    stat_config = Config.load("configs/context.json")

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(config.v_forward).to(device)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # dataset for statistics
    stat_dset = MotionDataset(train=True, config=stat_config)
    test_mean, test_std = stat_dset.test_statistics()
    test_mean, test_std = test_mean.to(device), test_std.to(device)

    # evaluation
    transition = [5, 15, 30, 45, 60, 90, 105, 120, 135, 150, 165, 180]
    print("================================================")
    print("|Transition   |  L2P  |  L2Q  |  L2T  |  NPSS  |")
    print("================================================")
    for t in transition:
        total_len = config.context_frames + t + 1
        eval_dict = {
            "L2P":  0,
            "L2Q":  0,
            "NPSS": 0,
            "L2T":  0,
        }
        GT_global_ps, GT_global_Qs, GT_trajs = [], [], []
        pred_global_ps, pred_global_Qs, pred_trajs = [], [], []
        for GT_motion in tqdm(dataloader, leave=False):
            """ 1. GT motion """
            GT_motion = GT_motion[:, :total_len, :-4].to(device) # exclude trajectory
            B, T, D = GT_motion.shape

            # trajectory        
            GT_traj = utils.get_trajectory(GT_motion, v_forward)

            # motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            GT_global_R6, GT_global_p = motionops.R6_fk(GT_local_R6, GT_root_p, skeleton)
            GT_global_Q = rotation.R6_to_Q(GT_global_R6)

            # add to list
            GT_global_ps.append(GT_global_p[:, config.context_frames:-1, :])
            GT_global_Qs.append(GT_global_Q[:, config.context_frames:-1, :])
            GT_trajs.append(GT_traj[:, config.context_frames:-1, :])

            """ 2. Interpolation """
            interp_motion = utils.get_interpolated_motion(GT_motion, config.context_frames)

            # trajectory
            interp_traj = utils.get_trajectory(interp_motion, v_forward)

            # motion
            interp_local_R6, interp_root_p = torch.split(interp_motion, [D-3, 3], dim=-1)
            interp_local_R6 = interp_local_R6.reshape(B, T, -1, 6)

            interp_global_R6, interp_global_p = motionops.R6_fk(interp_local_R6, interp_root_p, skeleton)
            interp_global_Q = rotation.R6_to_Q(interp_global_R6)

            # add to list
            pred_global_ps.append(interp_global_p[:, config.context_frames:-1, :])
            pred_global_Qs.append(interp_global_Q[:, config.context_frames:-1, :])
            pred_trajs.append(interp_traj[:, config.context_frames:-1, :])
        
        GT_global_p = torch.cat(GT_global_ps, dim=0).reshape(len(dataset), t, -1)
        GT_global_Q = torch.cat(GT_global_Qs, dim=0).reshape(len(dataset), t, -1)
        GT_traj     = torch.cat(GT_trajs, dim=0).reshape(len(dataset), t, -1)

        interp_global_p = torch.cat(pred_global_ps, dim=0).reshape(len(dataset), t, -1)
        interp_global_Q = torch.cat(pred_global_Qs, dim=0).reshape(len(dataset), t, -1)
        interp_traj     = torch.cat(pred_trajs, dim=0).reshape(len(dataset), t, -1)
        
        """ 3. Evaluation """
        # L2P
        GT_global_p = GT_global_p.transpose(1, 2)
        interp_global_p = interp_global_p.transpose(1, 2)
        norm_GT_p   = (GT_global_p - test_mean) / test_std
        norm_pred_p = (interp_global_p - test_mean) / test_std
        l2p = torch.mean(torch.sqrt(torch.sum((interp_global_p - GT_global_p)**2, dim=1))).item()

        # L2Q
        GT_global_Q     = utils.align_Q(GT_global_Q)
        interp_global_Q = utils.align_Q(interp_global_Q)
        l2q = torch.mean(torch.sqrt(torch.sum((interp_global_Q - GT_global_Q)**2, dim=(2, 3)))).item()

        # NPSS
        B, T, J, _ = GT_global_Q.shape
        GT_global_Q = GT_global_Q.reshape(B, T, -1)
        interp_global_Q = interp_global_Q.reshape(B, T, -1)
        npss = benchmark.NPSS(interp_global_Q, GT_global_Q)

        # L2T
        l2t = torch.mean(torch.sqrt(torch.sum((GT_traj - interp_traj)**2, dim=1))).item()
        
        print(f"Transition {t}: {l2p:.4f}\t{l2q:.4f}\t{l2t:.4f}\t{npss:.4f}")