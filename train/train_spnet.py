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

from utility.dataset import KeyframeDataset, GlobalKeyframeDataset
from utility.config import Config
from model.spnet import ScorePredictionNet
from utility import utils

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/spnet.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = GlobalKeyframeDataset(train=True, config=config)
    val_dataset = GlobalKeyframeDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics()
    motion_mean, motion_std = motion_mean[..., :-1], motion_std[..., :-1]
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = ScorePredictionNet(dataset.shape[-1] - 1, config).to(device)
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
        "score": 0,
    }

    # training
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_keyframe in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            """ 1. GT motion data """
            GT_keyframe = GT_keyframe[:, config.context_frames-1:, :]
            B, T, D = GT_keyframe.shape
            GT_keyframe = GT_keyframe.to(device)
            GT_motion, GT_kf_score = torch.split(GT_keyframe, [D-1, 1], dim=-1)

            """ 2. Train KF-VAE """
            # forward
            batch = (GT_motion - motion_mean) / motion_std
            batch = batch + torch.randn_like(batch) * 1e-2
            pred_kf_score = model.forward(batch)

            # loss
            loss = utils.recon_loss(pred_kf_score, GT_kf_score)

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            loss_dict["score"] += loss.item()

            """ 3. Log """
            if iter % config.log_interval == 0:
                utils.write_log(writer, loss_dict, config.log_interval, iter, train=True)
                utils.reset_log(loss_dict)
            
            """ 4. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "score":   0,
                    }
                    for GT_keyframe in val_dataloader:
                        # GT motion data
                        GT_keyframe = GT_keyframe[:, config.context_frames-1:, :]
                        B, T, D = GT_keyframe.shape
                        GT_keyframe = GT_keyframe.to(device)
                        GT_motion, GT_kf_score = torch.split(GT_keyframe, [D-1, 1], dim=-1)

                        # forward
                        batch = (GT_motion - motion_mean) / motion_std
                        pred_kf_score = model.forward(batch)

                        # loss
                        loss = utils.recon_loss(pred_kf_score, GT_kf_score)

                        # log
                        val_loss_dict["score"] += loss.item()

                # write and print log
                utils.write_log(writer, val_loss_dict, len(val_dataloader), iter, train=False)
                utils.reset_log(val_loss_dict)

                # train mode
                model.train()

            """ 5. Save checkpoint """
            if iter % config.save_interval == 0:
                utils.save_ckpt(model, optim, epoch, iter, config)#, scheduler=scheduler)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            # update iter
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    utils.save_ckpt(model, optim, epoch, iter, config)#, scheduler=scheduler)