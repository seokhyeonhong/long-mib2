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
from vis.visapp import TripletMotionApp
from model.keyframenet import KeyframeNet
from model.refinenet import RefineNet
from model.twostage import ContextTransformer, DetailTransformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset_config = Config.load("configs/dataset.json")
    kf_config  = Config.load("configs/keyframenet.json")
    ref_config = Config.load("configs/refinenet_nope.json")
    ctx_config = Config.load("configs/context.json")
    det_config = Config.load("configs/detail.json")

    # dataset - test
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=dset_config)
    dataloader = DataLoader(dataset, batch_size=ref_config.batch_size, shuffle=True)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(ref_config.v_forward).to(device)

    stat_dset = MotionDataset(train=True, config=ref_config)
    motion_mean, motion_std = stat_dset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = stat_dset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    test_mean, test_std = stat_dset.test_statistics()
    test_mean, test_std = test_mean.to(device), test_std.to(device)

    feet_ids = []
    for name in kf_config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    # model
    print("Initializing model...")
    kf_net = KeyframeNet(len(motion_mean), len(traj_mean), kf_config).to(device)
    utils.load_model(kf_net, kf_config)
    kf_net.eval()

    # ref_net = RefineNet(len(motion_mean), len(traj_mean), len(feet_ids), ref_config, local_attn=False).to(device)
    ref_net = RefineNet(len(motion_mean), len(traj_mean), len(feet_ids), ref_config, local_attn=ref_config.local_attn, use_pe=ref_config.use_pe).to(device)
    utils.load_model(ref_net, ref_config)
    ref_net.eval()

    ctx_model = ContextTransformer(len(motion_mean), ctx_config, len(traj_mean)).to(device)
    utils.load_model(ctx_model, ctx_config)
    ctx_model.eval()

    det_model = DetailTransformer(len(motion_mean), det_config, len(traj_mean)).to(device)
    utils.load_model(det_model, det_config)
    det_model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT motion """
            # GT_motion = GT_motion[:, :191]
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
            _, GT_global_p = utils.get_motion(GT_motion, skeleton)

            # Optional: Interpolate traj
            # GT_traj = utils.get_interpolated_trajectory(GT_traj, ref_config.context_frames)
            GT_traj[:, ref_config.context_frames:-1, 0:2] += torch.randn_like(GT_traj[:, ref_config.context_frames:-1, 0:2]) * 0.1

            # motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            """ 2. Forward KeyframeNet & RefineNet """
            # KeyframeNet
            motion = (GT_motion - motion_mean) / motion_std
            traj   = (GT_traj - traj_mean) / traj_std
            kf_motion, pred_score = kf_net.forward(motion, traj)
            kf_motion = kf_motion * motion_std + motion_mean

            # RefineNet
            ref_motions = []
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
                
                ref_motions.append(pred_motion)
            
            # concat predictions
            ref_motion = torch.cat(ref_motions, dim=0)

            # split
            ref_local_R6, ref_root_p = torch.split(ref_motion, [D-7, 3], dim=-1)
            ref_local_R = rotation.R6_to_R(ref_local_R6.reshape(B, T, -1, 6))
            _, ref_global_p = utils.get_motion(ref_motion, skeleton)

            """ 3. Forward ContextTransformer & DetailTransformer """
            motion = (GT_motion - motion_mean) / motion_std
            traj   = (GT_traj - traj_mean) / traj_std
            ctx_motion, mask = ctx_model.forward(motion, traj=traj, ratio_constrained=0.0)
            det_motion, _    = det_model.forward(ctx_motion, mask, traj=traj)
            det_motion = det_motion * motion_std + motion_mean

            # split
            det_local_R6, det_root_p = torch.split(det_motion, [D-7, 3], dim=-1)
            det_local_R = rotation.R6_to_R(det_local_R6.reshape(B, T, -1, 6))
            _, det_global_p = utils.get_motion(det_motion, skeleton)

            """ 4. Animation """
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            ref_local_R = ref_local_R.reshape(B*T, -1, 3, 3)
            ref_root_p = ref_root_p.reshape(B*T, -1)
            det_local_R = det_local_R.reshape(B*T, -1, 3, 3)
            det_root_p = det_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            ref_motion = Motion.from_torch(skeleton, ref_local_R, ref_root_p)
            det_motion = Motion.from_torch(skeleton, det_local_R, det_root_p)

            # total_kfs = copy.deepcopy(keyframes)
            # for b in range(1, B):
            #     for k in keyframes:
            #         total_kfs.append(k + b*T)
            
            """ 5. Evaluation """
            # L2P
            GT_global_p  = GT_global_p[:, ref_config.context_frames:-1]
            ref_global_p = ref_global_p[:, ref_config.context_frames:-1]
            det_global_p = det_global_p[:, ref_config.context_frames:-1]

            B, T = GT_global_p.shape[:2]
            GT_global_p = GT_global_p.reshape(B, T, -1).transpose(1, 2)
            ref_global_p = ref_global_p.reshape(B, T, -1).transpose(1, 2)
            det_global_p = det_global_p.reshape(B, T, -1).transpose(1, 2)

            norm_GT = (GT_global_p - test_mean) / test_std
            norm_ref = (ref_global_p - test_mean) / test_std
            norm_det = (det_global_p - test_mean) / test_std

            l2p_ref = torch.mean(torch.sqrt(torch.sum((norm_ref - norm_GT) ** 2, dim=1)), dim=-1)
            l2p_det = torch.mean(torch.sqrt(torch.sum((norm_det - norm_GT) ** 2, dim=1)), dim=-1)

            app_manager = AppManager()
            app = TripletMotionApp(GT_motion, ref_motion, det_motion, ybot.model(), T, l2ps=(l2p_ref, l2p_det))
            app_manager.run(app)