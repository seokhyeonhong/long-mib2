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
    # ref_config = Config.load("configs/refinenet.json")
    ref_config = Config.load("configs/refinenet_nope_res.json")
    abl_config1 = Config.load("configs/refinenet_nope_res_ablation1.json")
    abl_config2 = Config.load("configs/refinenet_nope_res_ablation2.json")

    # dataset - test
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=ref_config)
    dataloader = DataLoader(dataset, batch_size=ref_config.batch_size, shuffle=True)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(ref_config.v_forward).to(device)

    motion_mean, motion_std = dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = dataset.traj_statistics()
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

    abl_net1 = RefineNetResidual(len(motion_mean), len(traj_mean), len(feet_ids), abl_config1).to(device)
    utils.load_model(abl_net1, abl_config1, 600000)
    abl_net1.eval()

    abl_net2 = RefineNetResidual(len(motion_mean), len(traj_mean), len(feet_ids), abl_config2).to(device)
    utils.load_model(abl_net2, abl_config2, 600000)
    abl_net2.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT motion """
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)

            # Optional: Interpolate traj
            # GT_traj = utils.get_interpolated_trajectory(GT_traj, ref_config.context_frames)

            # motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            """ 2. Forward KeyframeNet """
            # normalize - forward - denormalize
            motion = (GT_motion - motion_mean) / motion_std
            traj   = (GT_traj - traj_mean) / traj_std
            kf_motion, pred_score = kf_net.forward(motion, traj)
            kf_motion = kf_motion * motion_std + motion_mean

            """ 3. Forward RefineNet """
            pred_motions = []
            abl1_motions = []
            abl2_motions = []
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

                # abl1
                abl1_motion, abl1_contact = abl_net1.forward(motion, traj[b:b+1], keyframes)
                abl1_motion = abl1_motion * motion_std + motion_mean
                abl1_motions.append(abl1_motion)

                # abl2
                abl2_motion, abl2_contact = abl_net2.forward(motion, traj[b:b+1], keyframes)
                abl2_motion = abl2_motion * motion_std + motion_mean
                abl2_motions.append(abl2_motion)
            
            # concat predictions
            pred_motion = torch.cat(pred_motions, dim=0)
            abl1_motion = torch.cat(abl1_motions, dim=0)
            abl2_motion = torch.cat(abl2_motions, dim=0)

            # split
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            abl1_local_R6, abl1_root_p = torch.split(abl1_motion, [D-7, 3], dim=-1)
            abl1_local_R = rotation.R6_to_R(abl1_local_R6.reshape(B, T, -1, 6))

            abl2_local_R6, abl2_root_p = torch.split(abl2_motion, [D-7, 3], dim=-1)
            abl2_local_R = rotation.R6_to_R(abl2_local_R6.reshape(B, T, -1, 6))

            """ 3. Animation """
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)

            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            abl1_local_R = abl1_local_R.reshape(B*T, -1, 3, 3)
            abl1_root_p = abl1_root_p.reshape(B*T, -1)

            abl2_local_R = abl2_local_R.reshape(B*T, -1, 3, 3)
            abl2_root_p = abl2_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)
            abl1_motion = Motion.from_torch(skeleton, abl1_local_R, abl1_root_p)
            abl2_motion = Motion.from_torch(skeleton, abl2_local_R, abl2_root_p)

            total_kfs = copy.deepcopy(keyframes)
            for b in range(1, B):
                for k in keyframes:
                    total_kfs.append(k + b*T)

            app_manager = AppManager()
            app = NeighborApp(GT_motion, pred_motion, [abl2_motion, abl1_motion], ybot.model(), T)
            app_manager.run(app)