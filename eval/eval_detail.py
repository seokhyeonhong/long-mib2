import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pymovis.ops import motionops, rotation, mathops

from utility.dataset import MotionDataset
from utility.config import Config
from utility import benchmark, utils
from model.twostage import ContextTransformer, DetailTransformer

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset_config = Config.load("configs/dataset.json")
    ctx_config = Config.load("configs/context.json")
    det_config = Config.load("configs/detail.json")

    # dataset - test
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=dset_config)
    dataloader = DataLoader(dataset, batch_size=ctx_config.batch_size, shuffle=False)

    # dataset - context
    ctx_dataset = MotionDataset(train=True, config=ctx_config)
    skeleton    = ctx_dataset.skeleton
    v_forward   = torch.from_numpy(ctx_config.v_forward).to(device)

    test_mean, test_std = ctx_dataset.test_statistics()
    test_mean, test_std = test_mean.to(device), test_std.to(device)

    motion_mean, motion_std = ctx_dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = ctx_dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    # model
    print("Initializing model...")
    # ctx_model = ContextTransformer(len(motion_mean), ctx_config).to(device)
    ctx_model = ContextTransformer(len(motion_mean), ctx_config, len(traj_mean)).to(device)
    utils.load_model(ctx_model, ctx_config)
    ctx_model.eval()

    # det_model = DetailTransformer(len(motion_mean), det_config).to(device)
    det_model = DetailTransformer(len(motion_mean), det_config, len(traj_mean)).to(device)
    utils.load_model(det_model, det_config, 300000)
    det_model.eval()

    # evaluation
    transition = [5, 15, 30, 60, 90, 120, 150, 180]
    for t in transition:
        total_len = ctx_config.context_frames + t + 1
            
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
                GT_global_ps.append(GT_global_p[:, ctx_config.context_frames:-1, :])
                GT_global_Qs.append(GT_global_Q[:, ctx_config.context_frames:-1, :])
                GT_trajs.append(GT_traj[:, ctx_config.context_frames:-1, :])

                """ 2. Forward """
                # forward
                motion = (GT_motion - motion_mean) / motion_std
                traj   = (GT_traj - traj_mean) / traj_std

                # use traj
                pred_motion, mask = ctx_model.forward(motion, traj=traj, ratio_constrained=0.0)
                pred_motion, _    = det_model.forward(pred_motion, mask, traj=traj)
                pred_motion = pred_motion * motion_std + motion_mean

                # no use traj
                # pred_motion, mask = ctx_model.forward(motion, ratio_constrained=0.0)
                # pred_motion, _    = det_model.forward(pred_motion, mask)
                # pred_motion = pred_motion * motion_std + motion_mean

                # trajectory
                pred_traj = utils.get_trajectory(pred_motion, v_forward)

                # motion
                pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
                pred_local_R6 = pred_local_R6.reshape(B, T, -1, 6)

                pred_global_R6, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)
                pred_global_Q = rotation.R6_to_Q(pred_global_R6)

                # add to list
                pred_global_ps.append(pred_global_p[:, ctx_config.context_frames:-1, :])
                pred_global_Qs.append(pred_global_Q[:, ctx_config.context_frames:-1, :])
                pred_trajs.append(pred_traj[:, ctx_config.context_frames:-1, :])
            
            GT_global_p = torch.cat(GT_global_ps, dim=0).reshape(len(dataset), t, -1)
            GT_global_Q = torch.cat(GT_global_Qs, dim=0).reshape(len(dataset), t, -1)
            GT_traj     = torch.cat(GT_trajs, dim=0).reshape(len(dataset), t, -1)

            pred_global_p = torch.cat(pred_global_ps, dim=0).reshape(len(dataset), t, -1)
            pred_global_Q = torch.cat(pred_global_Qs, dim=0).reshape(len(dataset), t, -1)
            pred_traj     = torch.cat(pred_trajs, dim=0).reshape(len(dataset), t, -1)
            
            """ 3. Evaluation """
            # L2P
            GT_global_p = GT_global_p.transpose(1, 2)
            pred_global_p = pred_global_p.transpose(1, 2)
            norm_GT_p   = (GT_global_p - test_mean) / test_std
            norm_pred_p = (pred_global_p - test_mean) / test_std
            l2p = torch.mean(torch.sqrt(torch.sum((pred_global_p - GT_global_p)**2, dim=1))).item()

            # L2Q
            B, T, D = GT_global_Q.shape
            GT_global_Q   = utils.remove_Q_discontinuities(GT_global_Q.reshape(B, T, -1, 4))
            pred_global_Q = utils.remove_Q_discontinuities(pred_global_Q.reshape(B, T, -1, 4))
            l2q = torch.mean(torch.sqrt(torch.sum((pred_global_Q - GT_global_Q)**2, dim=(2, 3)))).item()

            # NPSS
            B, T, J, _ = GT_global_Q.shape
            GT_global_Q = GT_global_Q.reshape(B, T, -1)
            pred_global_Q = pred_global_Q.reshape(B, T, -1)
            npss = benchmark.NPSS(pred_global_Q, GT_global_Q)

            # L2T
            l2t = torch.mean(torch.sqrt(torch.sum((GT_traj - pred_traj)**2, dim=1))).item()

            print("======Transition: {}======".format(t))
            print("L2P: {:.4f}, L2Q: {:.4f}, L2T: {:.4f}, NPSS: {:.4f}".format(l2p, l2q, l2t, npss))