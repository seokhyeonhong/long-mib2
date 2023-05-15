import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pymovis.ops import motionops, rotation, mathops

from utility.dataset import MotionDataset
from utility.config import Config
from utility import benchmark, utils
from model.keyframenet import KeyframeNet
from model.refinenet import RefineNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf_config  = Config.load("configs/keyframenet.json")
    ref_config = Config.load("configs/refinenet.json")

    # dataset - test
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=ref_config)
    dataloader = DataLoader(dataset, batch_size=ref_config.batch_size, shuffle=False)

    test_mean, test_std = dataset.test_statistics()
    test_mean, test_std = test_mean.to(device), test_std.to(device)

    # dataset - context
    ref_dataset = MotionDataset(train=True, config=ref_config)
    skeleton    = ref_dataset.skeleton
    v_forward   = torch.from_numpy(ref_config.v_forward).to(device)

    motion_mean, motion_std = ref_dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = ref_dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    # model
    print("Initializing model...")
    kf_net = KeyframeNet(len(motion_mean), len(traj_mean), kf_config).to(device)
    utils.load_model(kf_net, kf_config)
    kf_net.eval()

    ref_net = RefineNet(len(motion_mean), len(traj_mean), ref_config).to(device)
    utils.load_model(ref_net, ref_config)
    ref_net.eval()

    # evaluation
    transition = [5, 15, 30, 45]
    for t in transition:
        total_len = ref_config.context_frames + t + 1
            
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

                """ 2. Forward KeyframeNet """
                # normalize - forward - denormalize
                motion = (GT_motion - motion_mean) / motion_std
                traj   = (GT_traj   - traj_mean)   / traj_std
                kf_motion, pred_score = kf_net.forward(motion, traj)
                kf_motion = kf_motion * motion_std + motion_mean

                """ 3. Forward RefineNet """
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
                    
                    # forward
                    motion = ref_net.get_interpolated_motion(kf_motion[b:b+1], keyframes)
                    motion = (motion - motion_mean) / motion_std
                    pred_motion, pred_contact = ref_net.forward(motion, traj[b:b+1], keyframes)
                    pred_motion = pred_motion * motion_std + motion_mean
                    pred_motions.append(pred_motion)
                
                # concat predictions
                pred_motion = torch.cat(pred_motions, dim=0)

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