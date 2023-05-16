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
from vis.visapp import KeyframeApp
from model.keyframenet import KeyframeNet
from model.refinenet import RefineNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf_config  = Config.load("configs/keyframenet.json")
    # ref_config = Config.load("configs/refinenet.json")
    ref_config = Config.load("configs/refinenet_local.json")

    # dataset - test
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=ref_config)
    dataloader = DataLoader(dataset, batch_size=ref_config.batch_size, shuffle=True)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(ref_config.v_forward).to(device)

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
    utils.load_model(kf_net, kf_config, 100000)
    kf_net.eval()

    # ref_net = RefineNet(len(motion_mean), len(traj_mean), len(feet_ids), ref_config, local_attn=False).to(device)
    ref_net = RefineNet(len(motion_mean), len(traj_mean), len(feet_ids), ref_config).to(device)
    utils.load_model(ref_net, ref_config)
    ref_net.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT motion """
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)

            # motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            """ 2. Forward KeyframeNet """
            # normalize - forward - denormalize
            motion = (GT_motion - motion_mean) / motion_std
            traj   = (GT_traj   - traj_mean)   / traj_std
            kf_motion, pred_score = kf_net.forward(motion, traj)
            kf_motion = kf_motion * motion_std + motion_mean

            """ 3. Forward RefineNet """
            pred_motions = []
            interp_motions = []
            for b in range(B):
                # adaptive keyframe selection
                keyframes = [ref_config.context_frames - 1]
                transition_start = ref_config.context_frames
                while transition_start < T:
                    transition_end = min(transition_start + 30, T-1)
                    if transition_end == T-1:
                        keyframes.append(transition_end)
                        break

                    # top keyframe
                    top_keyframe = torch.topk(pred_score[b:b+1, transition_start+ref_config.min_transition:transition_end+1], 1, dim=1).indices + transition_start + ref_config.min_transition
                    top_keyframe = top_keyframe.item()
                    keyframes.append(top_keyframe)
                    transition_start = top_keyframe + 1
                
                # forward
                interp_motion = ref_net.get_interpolated_motion(kf_motion[b:b+1], keyframes)
                motion = (interp_motion - motion_mean) / motion_std
                pred_motion, pred_contact = ref_net.forward(motion, traj[b:b+1], keyframes)
                pred_motion = pred_motion * motion_std + motion_mean
                
                interp_motions.append(interp_motion)
                pred_motions.append(pred_motion)
            
            # concat predictions
            interp_motion = torch.cat(interp_motions, dim=0)
            pred_motion = torch.cat(pred_motions, dim=0)

            # split
            interp_local_R6, interp_root_p = torch.split(interp_motion, [D-7, 3], dim=-1)
            interp_local_R = rotation.R6_to_R(interp_local_R6.reshape(B, T, -1, 6))

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            """ 3. Animation """
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            interp_local_R = interp_local_R.reshape(B*T, -1, 3, 3)
            interp_root_p = interp_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            interp_motion = Motion.from_torch(skeleton, interp_local_R, interp_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            total_kfs = copy.deepcopy(keyframes)
            for b in range(1, B):
                for k in keyframes:
                    total_kfs.append(k + b*T)

            app_manager = AppManager()
            app = KeyframeApp(interp_motion, pred_motion, ybot.model(), T, total_kfs)
            app_manager.run(app)