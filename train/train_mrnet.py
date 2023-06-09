import sys
sys.path.append(".")

import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
from tqdm import tqdm

from pymovis.utils import util, torchconst
from pymovis.ops import motionops, rotation, mathops

from utility.dataset import MotionDataset
from utility.config import Config
from model.mrnet import MotionRefineNet
from utility import utils

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/mrnet.json")
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
    
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = MotionRefineNet(dataset.shape[-1] - 4, 4, config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = utils.get_noam_scheduler(config, optim)
    init_epoch, iter = utils.load_latest_ckpt(model, optim, config)#, scheduler=scheduler)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # loss dict
    loss_dict = {
        "total":   0,
        "pose":    0,
        "traj":    0,
        "vel":     0,
        "contact": 0,
        "foot":    0,
    }

    # training
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            """ 1. Random transiiton length """
            transition = random.randint(config.min_transition, config.max_transition)
            T = config.context_frames + transition + 1
            GT_motion = GT_motion[:, :T, :].to(device)

            """ 2. GT motion data """
            B, T, D = GT_motion.shape
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
            GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)
            GT_feet_v, GT_contact = utils.get_velocity_and_contact(GT_global_p, feet_ids, config.contact_vel_threshold)

            """ 3. Interpolated motion data & mask """
            interp_motion = utils.get_interpolated_motion(GT_motion, config.context_frames)

            mask = torch.ones(B, T, 1, device=device, dtype=torch.float32)
            mask[:, config.context_frames:-1, :] = 0

            """ 4. Train MotionRefineNet """
            # forward
            motion = (interp_motion - motion_mean) / motion_std
            traj   = (GT_traj - traj_mean) / traj_std
            pred_motion, pred_contact = model.forward(motion, mask, traj)

            # predicted motion
            pred_motion = pred_motion * motion_std + motion_mean
            pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)
            pred_feet_v, _ = utils.get_velocity_and_contact(pred_global_p, feet_ids, config.contact_vel_threshold)

            # loss
            loss_pose    = config.weight_pose    * (utils.recon_loss(pred_local_R6, GT_local_R6) + utils.recon_loss(pred_global_p, GT_global_p))
            loss_traj    = config.weight_traj    * (utils.traj_loss(pred_traj, GT_traj))
            loss_vel     = config.weight_vel     * (utils.recon_loss(pred_global_p[:, 1:] - pred_global_p[:, :-1], GT_global_p[:, 1:] - GT_global_p[:, :-1]))
            loss_contact = config.weight_contact * (utils.recon_loss(pred_contact, GT_contact))
            loss_foot    = config.weight_foot    * (utils.foot_loss(pred_feet_v, pred_contact.detach()))
            loss         = loss_pose + loss_traj + loss_vel + loss_contact + loss_foot

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            loss_dict["total"]   += loss.item()
            loss_dict["pose"]    += loss_pose.item()
            loss_dict["traj"]    += loss_traj.item()
            loss_dict["vel"]     += loss_vel.item()
            loss_dict["contact"] += loss_contact.item()
            loss_dict["foot"]    += loss_foot.item()

            """ 5. Log """
            if iter % config.log_interval == 0:
                utils.write_log(writer, loss_dict, config.log_interval, iter, train=True)
                utils.reset_log(loss_dict)
            
            """ 6. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total":   0,
                        "pose":    0,
                        "traj":    0,
                        "vel":     0,
                        "contact": 0,
                        "foot":    0,
                    }

                    for GT_motion in val_dataloader:
                        """ 1. Max transiiton length """
                        T = config.context_frames + config.max_transition + 1
                        GT_motion = GT_motion[:, :T, :].to(device)

                        """ 2. GT motion data """
                        B, T, D = GT_motion.shape
                        GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
                        GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)
                        GT_feet_v, GT_contact = utils.get_velocity_and_contact(GT_global_p, feet_ids, config.contact_vel_threshold)

                        """ 3. Interpolated motion data & mask """
                        interp_motion = utils.get_interpolated_motion(GT_motion, config.context_frames)

                        mask = torch.ones(B, T, 1, device=device, dtype=torch.float32)
                        mask[:, config.context_frames:-1, :] = 0

                        """ 4. Train MotionRefineNet """
                        # forward
                        motion = (interp_motion - motion_mean) / motion_std
                        traj   = (GT_traj - traj_mean) / traj_std
                        pred_motion, pred_contact = model.forward(motion, mask, traj)

                        # predicted motion
                        pred_motion = pred_motion * motion_std + motion_mean
                        pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)
                        pred_feet_v, _ = utils.get_velocity_and_contact(pred_global_p, feet_ids, config.contact_vel_threshold)

                        # loss
                        loss_pose    = config.weight_pose    * (utils.recon_loss(pred_local_R6, GT_local_R6) + utils.recon_loss(pred_global_p, GT_global_p))
                        loss_traj    = config.weight_traj    * (utils.traj_loss(pred_traj, GT_traj))
                        loss_vel     = config.weight_vel     * (utils.recon_loss(pred_global_p[:, 1:] - pred_global_p[:, :-1], GT_global_p[:, 1:] - GT_global_p[:, :-1]))
                        loss_contact = config.weight_contact * (utils.recon_loss(pred_contact, GT_contact))
                        loss_foot    = config.weight_foot    * (utils.foot_loss(pred_feet_v, pred_contact.detach()))
                        loss         = loss_pose + loss_traj + loss_vel + loss_contact + loss_foot

                        # log
                        val_loss_dict["total"]   += loss.item()
                        val_loss_dict["pose"]    += loss_pose.item()
                        val_loss_dict["traj"]    += loss_traj.item()
                        val_loss_dict["vel"]     += loss_vel.item()
                        val_loss_dict["contact"] += loss_contact.item()
                        val_loss_dict["foot"]    += loss_foot.item()

                # write and print log
                utils.write_log(writer, val_loss_dict, len(val_dataloader), iter, train=False)
                utils.reset_log(val_loss_dict)

                # train mode
                model.train()

            """ 5. Save checkpoint """
            if iter % config.save_interval == 0:
                utils.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            # update iter
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    utils.save_ckpt(model, optim, epoch, iter, config)