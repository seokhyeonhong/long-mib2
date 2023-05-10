import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pymovis.ops import motionops, rotation, mathops

from utility.dataset import MotionDataset
from utility.config import Config
from utility import benchmark, utils
from model.twostage import ContextTransformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config     = Config.load("configs/dataset.json")
    ctx_config = Config.load("configs/context.json")

    # dataset - test
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    test_mean, test_std = dataset.test_statistics()
    test_mean, test_std = test_mean.to(device), test_std.to(device)

    # dataset - context
    ctx_dataset = MotionDataset(train=True, config=ctx_config)
    skeleton    = ctx_dataset.skeleton
    v_forward   = torch.from_numpy(ctx_config.v_forward).to(device)

    motion_mean, motion_std = ctx_dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = ctx_dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)


    # model
    print("Initializing model...")
    ctx_model = ContextTransformer(len(motion_mean), len(traj_mean), ctx_config).to(device)
    utils.load_model(ctx_model, ctx_config)
    ctx_model.eval()

    # evaluation
    transition = [5, 15, 30, 45]
    for t in transition:
        total_len = config.context_frames + t + 1
            
        GT_global_ps, GT_global_Qs, GT_trajs = [], [], []
        pred_global_ps, pred_global_Qs, pred_trajs = [], [], []
        with torch.no_grad():
            for GT_motion in tqdm(dataloader):
                """ 1. GT motion """
                GT_motion = GT_motion[:, :total_len].to(device)
                B, T, D = GT_motion.shape
                GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)

                # motion
                GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)
                GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
                GT_global_R6, GT_global_p = motionops.R6_fk(GT_local_R6, GT_root_p, skeleton)
                GT_global_Q = rotation.R6_to_Q(GT_global_R6)

                # add to list
                GT_global_ps.append(GT_global_p)
                GT_global_Qs.append(GT_global_Q)
                GT_trajs.append(GT_traj)

                """ 2. Forward """
                # forward
                motion = (GT_motion - motion_mean) / motion_std
                traj   = (GT_traj - traj_mean) / traj_std
                pred_motion, mask = ctx_model.forward(motion, traj, ratio_constrained=0.0)
                pred_motion = pred_motion * motion_std + motion_mean

                # trajectory
                pred_traj = utils.get_trajectory(pred_motion, v_forward)

                # motion
                pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
                pred_local_R6 = pred_local_R6.reshape(B, T, -1, 6)

                pred_global_R6, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)
                pred_global_Q = rotation.R6_to_Q(pred_global_R6)

                # add to list
                pred_global_ps.append(pred_global_p)
                pred_global_Qs.append(pred_global_Q)
                pred_trajs.append(pred_traj)
            
            GT_global_p = torch.cat(GT_global_ps, dim=0).reshape(len(dataset), total_len, -1)
            GT_global_Q = torch.cat(GT_global_Qs, dim=0).reshape(len(dataset), total_len, -1)
            GT_traj     = torch.cat(GT_trajs, dim=0).reshape(len(dataset), total_len, -1)

            pred_global_p = torch.cat(pred_global_ps, dim=0).reshape(len(dataset), total_len, -1)
            pred_global_Q = torch.cat(pred_global_Qs, dim=0).reshape(len(dataset), total_len, -1)
            pred_traj     = torch.cat(pred_trajs, dim=0).reshape(len(dataset), total_len, -1)
            
            """ 3. Evaluation """
            # L2P
            norm_GT_p   = (GT_global_p - test_mean) / test_std
            norm_pred_p = (pred_global_p - test_mean) / test_std
            l2p = torch.mean(torch.norm(norm_GT_p - norm_pred_p, dim=-1)).item()

            # L2Q
            l2q = torch.mean(torch.norm(GT_global_Q - pred_global_Q, dim=-1)).item()

            # NPSS
            npss = benchmark.NPSS(pred_global_Q, GT_global_Q)

            # L2T
            l2t = torch.mean(torch.norm(GT_traj - pred_traj, dim=-1)).item()

            print("======Transition: {}======".format(t))
            print("L2P: {:.4f}, L2Q: {:.4f}, NPSS: {:.4f}, L2T: {:.4f}".format(l2p, l2q, npss, l2t))