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
from model.mpvae import MotionPredictionVAE
from utility import utils

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/mpvae.json")
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
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = MotionPredictionVAE(dataset.shape[-1] - 4, 4, config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    init_epoch, iter = utils.load_latest_ckpt(model, optim, config)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # loss dict
    loss_dict = {
        "total":  0,
        "pose":   0,
        "traj":   0,
        "smooth": 0,
        "kl":     0,
    }

    # training
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            """ 1. GT motion data """
            B, T, D = GT_motion.shape
            GT_motion = GT_motion.to(device)
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
            GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)

            """ 2. Train KF-VAE """
            # forward
            motion = (GT_motion - motion_mean) / motion_std
            traj   = (GT_traj - traj_mean) / traj_std
            pred_motion, pred_mean, pred_logvar = model.forward(motion, traj)

            # prediction
            pred_motion = pred_motion * motion_std + motion_mean
            pred_local_R6, pred_global_p, pred_traj  = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)

            # loss
            loss_pose   = config.weight_pose * (utils.recon_loss(pred_local_R6, GT_local_R6) + utils.recon_loss(pred_global_p, GT_global_p))
            loss_traj   = config.weight_traj * utils.traj_loss(pred_traj, GT_traj)
            loss_smooth = config.weight_smooth * (utils.smooth_loss(pred_local_R6) + utils.smooth_loss(pred_global_p))
            loss_kl     = config.weight_kl * utils.kl_loss(pred_mean, pred_logvar)
            loss        = loss_pose + loss_traj + loss_smooth + loss_kl

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            loss_dict["total"]  += loss.item()
            loss_dict["pose"]   += loss_pose.item()
            loss_dict["traj"]   += loss_traj.item()
            loss_dict["smooth"] += loss_smooth.item()
            loss_dict["kl"]     += loss_kl.item()

            """ 3. Log """
            if iter % config.log_interval == 0:
                utils.write_log(writer, loss_dict, config.log_interval, iter, train=True)
                utils.reset_log(loss_dict)
            
            """ 4. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total":  0,
                        "pose":   0,
                        "traj":   0,
                        "smooth": 0,
                    }
                    for GT_motion in val_dataloader:
                        # GT motion data
                        B, T, D = GT_motion.shape
                        GT_motion = GT_motion.to(device)
                        GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
                        GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)

                        # forward
                        motion = (GT_motion - motion_mean) / motion_std
                        traj   = (GT_traj - traj_mean) / traj_std
                        pred_motion = model.sample(motion, traj)

                        # prediction
                        pred_motion = pred_motion * motion_std + motion_mean
                        pred_local_R6, pred_global_p, pred_traj  = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)

                        # loss
                        loss_pose   = config.weight_pose * (utils.recon_loss(pred_local_R6, GT_local_R6) + utils.recon_loss(pred_global_p, GT_global_p))
                        loss_traj   = config.weight_traj * utils.traj_loss(pred_traj, GT_traj)
                        loss_smooth = config.weight_smooth * (utils.smooth_loss(pred_local_R6) + utils.smooth_loss(pred_global_p))
                        loss        = loss_pose + loss_traj + loss_smooth

                        # log
                        val_loss_dict["total"]  += loss.item()
                        val_loss_dict["pose"]   += loss_pose.item()
                        val_loss_dict["traj"]   += loss_traj.item()
                        val_loss_dict["smooth"] += loss_smooth.item()

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