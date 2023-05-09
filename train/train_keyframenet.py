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

from pymovis.utils import util

from utility.dataset import KeyframeDataset
from utility.config import Config
from model.keyframenet import KeyframeNet
from utility import utils

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframenet.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = KeyframeDataset(train=True,  config=config)
    val_dataset = KeyframeDataset(train=False, config=config)
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
    model = KeyframeNet(len(motion_mean), len(traj_mean), config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
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
        "score":  0,
        "traj":   0,
    }

    # training
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_keyframe in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            """ 1. GT data """
            B, T, D = GT_keyframe.shape
            GT_keyframe = GT_keyframe.to(device)
            GT_motion, GT_traj, GT_score = torch.split(GT_keyframe, [D-5, 4, 1], dim=-1)

            GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)
            
            """ 2. Forward """
            # normalize - forward - denormalize
            motion = (GT_motion - motion_mean) / motion_std
            traj   = (GT_traj   - traj_mean)   / traj_std
            pred_motion, pred_score = model.forward(motion, traj)
            pred_motion = pred_motion * motion_std + motion_mean

            # predicted motion
            pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)

            """ 3. Loss & Backward """
            # loss
            loss_pose  = config.weight_pose * (utils.recon_loss(pred_local_R6, GT_local_R6) + utils.recon_loss(pred_global_p, GT_global_p))
            loss_score = config.weight_score * utils.recon_loss(pred_score, GT_score)
            loss_traj  = config.weight_traj * (utils.traj_loss(pred_traj, GT_traj))

            loss = loss_pose + loss_score + loss_traj

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            """ 4. Log """
            loss_dict["total"] += loss.item()
            loss_dict["pose"]  += loss_pose.item()
            loss_dict["score"] += loss_score.item()
            loss_dict["traj"]  += loss_traj.item()

            if iter % config.log_interval == 0:
                utils.write_log(writer, loss_dict, config.log_interval, iter, train=True)
                utils.reset_log(loss_dict)
            
            """ 5. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total": 0,
                        "pose":  0,
                        "score": 0,
                        "traj":  0,
                    }
                    for GT_keyframe in tqdm(val_dataloader, desc=f"Validation", leave=False):
                        """ 1. GT data """
                        B, T, D = GT_keyframe.shape
                        GT_keyframe = GT_keyframe.to(device)
                        GT_motion, GT_traj, GT_score = torch.split(GT_keyframe, [D-5, 4, 1], dim=-1)

                        GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)
                        
                        """ 2. Forward """
                        # normalize - forward - denormalize
                        motion = (GT_motion - motion_mean) / motion_std
                        traj   = (GT_traj   - traj_mean)   / traj_std
                        pred_motion, pred_score = model.forward(motion, traj)
                        pred_motion = pred_motion * motion_std + motion_mean

                        # predicted motion
                        pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)

                        """ 3. Loss """
                        # loss
                        loss_pose  = config.weight_pose * (utils.recon_loss(pred_local_R6, GT_local_R6) + utils.recon_loss(pred_global_p, GT_global_p))
                        loss_score = config.weight_score * utils.recon_loss(pred_score, GT_score)
                        loss_traj  = config.weight_traj * (utils.traj_loss(pred_traj, GT_traj))

                        loss = loss_pose + loss_score + loss_traj

                        # log
                        val_loss_dict["total"]  += loss.item()
                        val_loss_dict["pose"]   += loss_pose.item()
                        val_loss_dict["score"]  += loss_score.item()
                        val_loss_dict["traj"]   += loss_traj.item()

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