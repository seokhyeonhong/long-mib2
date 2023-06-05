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
from vis.visapp import NeighborApp
from model.keyframenet import KeyframeNet
from model.refinenet import RefineNetResidual

if __name__ == "__main__":
    util.seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf_config  = Config.load("configs/keyframenet.json")
    ref_config = Config.load("configs/refinenet_nope_res.json")

    # dataset - test
    print("Loading dataset...")
    train_dset   = MotionDataset(train=True,  config=ref_config)
    test_dset    = MotionDataset(train=False, config=ref_config)
    print(train_dset.features.shape)
    print(test_dset.features.shape)
    breakpoint()

    train_loader = DataLoader(train_dset, batch_size=ref_config.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dset,  batch_size=ref_config.batch_size, shuffle=True)
    
    skeleton     = train_dset.skeleton
    v_forward    = torch.from_numpy(ref_config.v_forward).to(device)

    motion_mean, motion_std = train_dset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = train_dset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    feet_ids = []
    for name in kf_config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    # model
    print("Initializing model...")
    kf_net = KeyframeNet(len(motion_mean), len(traj_mean), kf_config).to(device)
    utils.load_model(kf_net, kf_config)
    kf_net.eval()

    ref_net = RefineNetResidual(len(motion_mean), len(traj_mean), len(feet_ids), ref_config).to(device)
    utils.load_model(ref_net, ref_config, 600000)
    ref_net.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # construct database from training set
    motion_db, traj_db = [], []
    vis_db = []
    for train_data in train_loader:
        B, T, D = train_data.shape
        train_data = train_data.to(device)

        motion, traj = torch.split(train_data, [D-4, 4], dim=-1)
        vis_db.append(motion)

        cond_motion = torch.cat([motion[:, 0:ref_config.context_frames, :], motion[:, -1:, :]], dim=1)
        cond_motion = cond_motion.reshape(B, -1)
        traj        = traj.reshape(B, -1)
        motion_db.append(cond_motion)
        traj_db.append(traj)

    motion_db    = torch.cat(motion_db, dim=0)
    traj_db      = torch.cat(traj_db, dim=0)
    vis_db       = torch.cat(vis_db, dim=0)

    # loop
    with torch.no_grad():
        for GT_motion in tqdm(test_loader):
            """ 1. GT motion """
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)

            # Optional: Interpolate traj
            GT_traj = utils.get_interpolated_trajectory(GT_traj, ref_config.context_frames)#, min_scale=1.5, max_scale=1.5)
            GT_motion[:, -1, (-3, -1)] = GT_traj[:, -1, 0:2]

            # motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            """ 2. Find the Nearest Neighbors from Database """
            query_motion = torch.cat([GT_motion[:, 0:ref_config.context_frames, :], GT_motion[:, -1:, :]], dim=1)
            query_motion = query_motion.reshape(B, -1)
            query_traj   = GT_traj.reshape(B, -1)

            k = 5
            motion_dist = torch.cdist(query_motion, motion_db, p=2)
            traj_dist   = torch.cdist(query_traj, traj_db, p=2)
            _, nn_idx = torch.topk(motion_dist + traj_dist, k, dim=1, largest=False)
            nn_motion = torch.index_select(vis_db, 0, nn_idx.reshape(-1))
            nn_dist   = (motion_dist + traj_dist).gather(dim=1, index=nn_idx)
            nn_dist   = torch.mean(nn_dist, dim=-1)
            print(nn_dist)

            """ 3. Forward KeyframeNet & RefineNet """
            # KeyframeNet
            motion = (GT_motion - motion_mean) / motion_std
            traj   = (GT_traj - traj_mean) / traj_std
            kf_motion, pred_score = kf_net.forward(motion, traj)
            kf_motion = kf_motion * motion_std + motion_mean

            # RefineNet
            pred_motions = []
            for b in range(B):
                # adaptive keyframe selection
                keyframes = [ref_config.context_frames - 1]
                transition_start = ref_config.context_frames
                while transition_start < T:
                    transition_end = min(transition_start + ref_config.max_transition, T-1)
                    if transition_end == T-1:
                        keyframes.append(transition_end)
                        break

                    # top keyframe
                    top_keyframe = torch.topk(pred_score[b:b+1, transition_start+ref_config.min_transition:transition_end+1], 1, dim=1).indices + transition_start + ref_config.min_transition
                    top_keyframe = top_keyframe.item()
                    keyframes.append(top_keyframe)
                    transition_start = top_keyframe + 1
                
                # forward - interp
                interp_motion = ref_net.get_interpolated_motion(kf_motion[b:b+1], keyframes)
                motion = (interp_motion - motion_mean) / motion_std

                # forward - nointerp
                # motion = (kf_motion[b:b+1] - motion_mean) / motion_std

                pred_motion, pred_contact = ref_net.forward(motion, traj[b:b+1], keyframes)
                pred_motion = pred_motion * motion_std + motion_mean
                pred_motions.append(pred_motion)
            
            # concat predictions
            pred_motion = torch.cat(pred_motions, dim=0)

            # split
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            nn_local_R6, nn_root_p = torch.split(nn_motion, [D-7, 3], dim=-1)
            nn_local_R = rotation.R6_to_R(nn_local_R6.reshape(B*k, T, -1, 6))

            """ 3. Animation """
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)

            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            nn_local_R = nn_local_R.reshape(B, k, T, -1, 3, 3).transpose(0, 1).reshape(k, B*T, -1, 3, 3)
            nn_root_p = nn_root_p.reshape(B, k, T, -1).transpose(0, 1).reshape(k, B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)
            nn_motions = []
            for k_ in range(k):
                nn_motion = Motion.from_torch(skeleton, nn_local_R[k_], nn_root_p[k_])
                nn_motions.append(nn_motion)

            total_kfs = copy.deepcopy(keyframes)
            for b in range(1, B):
                for k in keyframes:
                    total_kfs.append(k + b*T)

            app_manager = AppManager()
            app = NeighborApp(GT_motion, pred_motion, nn_motions, ybot.model(), T, text=nn_dist.cpu().numpy())
            app_manager.run(app)