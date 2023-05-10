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
from pymovis.ops import motionops

from utility.dataset import MotionDataset
from utility.config import Config
from model.twostage import ContextTransformer
from utility import utils

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=True, config=config)
    val_dataset = MotionDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.motion_statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    traj_mean, traj_std = dataset.traj_statistics(dim=(0, 1))
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = ContextTransformer(len(motion_mean), len(traj_mean), config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
    init_epoch, iter = utils.load_latest_ckpt(model, optim, config)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # training loop
    loss_dict = {
        "total":  0,
        "rot":    0,
        "pos":    0,
        "smooth": 0,
        "traj":   0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            # GT
            transition_frames = random.randint(config.min_transition, config.max_transition)
            T = config.context_frames + transition_frames + 1
            GT_motion = GT_motion[:, :T].to(device)

            B, T, D = GT_motion.shape
            GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
            GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)

            # forward
            motion_batch = (GT_motion - motion_mean) / motion_std
            traj_batch   = (GT_traj - traj_mean) / traj_std
            pred_motion, _ = model.forward(motion_batch, traj_batch)
            pred_motion = pred_motion * motion_std + motion_mean
            pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)
            
            # loss
            loss_rot    = config.weight_rot    * utils.recon_loss(pred_local_R6[:, config.context_frames:-1], GT_local_R6[:, config.context_frames:-1])
            loss_pos    = config.weight_pos    * utils.recon_loss(pred_global_p[:, config.context_frames:-1], GT_global_p[:, config.context_frames:-1])
            loss_smooth = config.weight_smooth * utils.smooth_loss(pred_global_p[:, config.context_frames-1:])
            loss_traj   = config.weight_traj   * utils.traj_loss(pred_traj[:, config.context_frames:-1], GT_traj[:, config.context_frames:-1])
            loss        = loss_rot + loss_pos + loss_smooth + loss_traj

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            loss_dict["total"]  += loss.item()
            loss_dict["rot"]    += loss_rot.item()
            loss_dict["pos"]    += loss_pos.item()
            loss_dict["smooth"] += loss_smooth.item()
            loss_dict["traj"]   += loss_traj.item()

            if iter % config.log_interval == 0:
                utils.write_log(writer, loss_dict, config.log_interval, iter, elapsed=time.perf_counter() - start_time, train=True)
                utils.reset_log(loss_dict)
            
            # validation
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total":  0,
                        "rot":    0,
                        "pos":    0,
                        "smooth": 0,
                        "traj":   0,
                    }
                    for GT_motion in val_dataloader:
                        # GT
                        T = config.context_frames + config.max_transition + 1
                        GT_motion = GT_motion[:, :T].to(device)

                        B, T, D = GT_motion.shape
                        GT_motion, GT_traj = torch.split(GT_motion, [D-4, 4], dim=-1)
                        GT_local_R6, GT_global_p = utils.get_motion(GT_motion, skeleton)

                        # forward
                        motion_batch = (GT_motion - motion_mean) / motion_std
                        traj_batch   = (GT_traj - traj_mean) / traj_std
                        pred_motion, _ = model.forward(motion_batch, traj_batch)
                        pred_motion = pred_motion * motion_std + motion_mean
                        pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)

                        # loss
                        loss_rot    = config.weight_rot    * utils.recon_loss(pred_local_R6[:, config.context_frames:-1], GT_local_R6[:, config.context_frames:-1])
                        loss_pos    = config.weight_pos    * utils.recon_loss(pred_global_p[:, config.context_frames:-1], GT_global_p[:, config.context_frames:-1])
                        loss_smooth = config.weight_smooth * utils.smooth_loss(pred_global_p[:, config.context_frames-1:])
                        loss_traj   = config.weight_traj   * utils.traj_loss(pred_traj[:, config.context_frames:-1], GT_traj[:, config.context_frames:-1])
                        loss        = loss_rot + loss_pos + loss_smooth + loss_traj

                        # log
                        val_loss_dict["total"]  += loss.item()
                        val_loss_dict["rot"]    += loss_rot.item()
                        val_loss_dict["pos"]    += loss_pos.item()
                        val_loss_dict["smooth"] += loss_smooth.item()
                        val_loss_dict["traj"]   += loss_traj.item()

                    utils.write_log(writer, val_loss_dict, len(val_dataloader), iter, elapsed=time.perf_counter() - start_time, train=False)
                    utils.reset_log(val_loss_dict)

                model.train()

            # save
            if iter % config.save_interval == 0:
                utils.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    utils.save_ckpt(model, optim, epoch, iter, config)