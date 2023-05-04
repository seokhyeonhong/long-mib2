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
from model.mpvae import MotionPredictionVAE
from model.mrnet import MotionRefineNet

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/longer_dataset.json")
    mpvae_cfg = Config.load("configs/mpvae.json")
    mrnet_cfg = Config.load("configs/mrnet.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    # temp_dataset: for statistics used in training
    temp_dataset = MotionDataset(train=True, config=mpvae_cfg)
    motion_mean, motion_std = temp_dataset.motion_statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = temp_dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    mpvae = MotionPredictionVAE(len(motion_mean), len(traj_mean), mpvae_cfg).to(device)
    utils.load_model(mpvae, mpvae_cfg)
    mpvae.eval()

    mrnet = MotionRefineNet(len(motion_mean), len(traj_mean), mrnet_cfg).to(device)
    utils.load_model(mrnet, mrnet_cfg)
    mrnet.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. GT motion data """
            # GT_motion = GT_motion[:, :config.max_transition+config.context_frames+1]
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)

            GT_motion = utils.get_interpolated_motion(GT_motion, config.context_frames)
            GT_traj = utils.get_interpolated_trajectory(GT_traj, config.context_frames)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-7, 3], dim=-1)

            """ 2. For each batch """
            results = []
            keyframes = []
            for b in range(B):
                batch = GT_motion[b:b+1].clone()
                start_frame = 0
                end_frame = 0
                while end_frame < T:
                    ctx_frame = start_frame + config.context_frames - 1
                    end_frame = min(start_frame + config.context_frames + config.max_transition + 1, T)

                    """ 2-1. Align MP-VAE input to origin and forward at the context frame """
                    R_diff, root_p_diff = utils.get_align_Rp(batch, ctx_frame, v_forward)
                    motion_batch = utils.align_motion(batch, R_diff, root_p_diff)
                    traj_batch   = utils.get_trajectory(motion_batch, v_forward)
                    breakpoint()

                    """ 2-2. Generate motion from MP-VAE """
                    motion_batch = (motion_batch - motion_mean) / motion_std
                    traj_batch   = (traj_batch - traj_mean) / traj_std
                    pred_motion = mpvae.sample(motion_batch[:, start_frame:end_frame], traj_batch[:, start_frame:end_frame])
                    pred_motion = pred_motion * motion_std + motion_mean

                    """ 2-3. Sample keyframe """
                    if end_frame != T:
                        # interpolation
                        interp_motion = utils.get_interpolated_motion(pred_motion, config.context_frames)

                        # measure differences
                        _, pred_global_p = utils.get_motion(pred_motion, skeleton)
                        _, interp_global_p = utils.get_motion(interp_motion, skeleton)
                        diff = torch.norm(pred_global_p - interp_global_p, dim=-1) # (1, T, J)
                        diff = torch.sum(diff, dim=-1) # (1, T)
                        diff = diff[:, config.context_frames+5:] # (1, T-ctx_frame-5)

                        # get keyframe (= max diff frame)
                        keyframe = torch.argmax(diff, dim=1).item() + config.context_frames + 5
                        keyframes.append(start_frame + keyframe + b * T)
                    else:
                        # set final frame as keyframe
                        keyframe = end_frame - start_frame - 1
                        pred_motion[:, -1] = motion_batch[:, -1] * motion_std + motion_mean

                    # split
                    pred_motion = pred_motion[:, :keyframe+1]

                    """ 2-4. Interpolation and refine """
                    # interpolation & mask
                    interp_motion = utils.get_interpolated_motion(pred_motion, config.context_frames) # (1, keyframe+1, D)

                    mask = torch.ones(1, keyframe+1, 1, device=device, dtype=torch.float32)
                    mask[:, config.context_frames:-1, :] = 0

                    # refine
                    motion_batch = (interp_motion - motion_mean) / motion_std
                    traj_batch   = traj_batch[:, start_frame:start_frame+keyframe+1]
                    pred_motion, pred_contact = mrnet.forward(motion_batch, mask, traj_batch)
                    pred_motion  = pred_motion * motion_std + motion_mean

                    """ 2-5. Restore to the original position and orientation """
                    pred_motion = utils.restore_motion(pred_motion, R_diff, root_p_diff)

                    """ 2-6. Update """
                    batch[:, start_frame:start_frame+keyframe+1] = pred_motion
                    start_frame += (keyframe + 1) - config.context_frames

                results.append(batch)
            
            """ 3. Save """
            pred_motion = torch.cat(results, dim=0)
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)

            """ 4. Animation """
            # animation
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6)).reshape(B*T, -1, 3, 3)
            GT_root_p  = GT_root_p.reshape(B*T, -1)

            pred_local_R = rotation.R6_to_R(pred_motion[:, :, :-3].reshape(B, T, -1, 6)).reshape(B*T, -1, 3, 3)
            pred_root_p  = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            app = KeyframeApp(GT_motion, pred_motion, ybot.model(), T, keyframes)
            app_manager.run(app)