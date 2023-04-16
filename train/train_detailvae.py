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
from model.vae import VAE
from utility import trainutil, testutil

if __name__ == "__main__":
    # initial settings with all possible gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/detail_vae.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=True, config=config)
    val_dataset = MotionDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)
    
    # mean and std
    motion_mean, motion_std = dataset.statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    # dataset loader
    dataloader     = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # foot joint indices
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    # model
    print("Initializing model...")
    ctx_config = Config.load("configs/context_vae.json")
    ctx_model = VAE(dataset.shape[-1], ctx_config, is_context=True).to(device)
    testutil.load_model(ctx_model, ctx_config)
    ctx_model.eval()

    det_model = VAE(dataset.shape[-1], config, is_context=False).to(device)
    optim = torch.optim.Adam(det_model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    init_epoch, iter = trainutil.load_latest_ckpt(det_model, optim, config)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # training loop
    loss_dict = {
        "total":   0,
        "pose":    0,
        "traj":    0,
        "contact": 0,
        "foot":    0,
        "kl":      0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            transition = random.randint(config.min_transition, config.max_transition)
            T = config.context_frames + transition + 1
            GT_motion = GT_motion[:, :T, :].to(device)
            B, T, D = GT_motion.shape

            """ 1. GT motion data """
            # motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            _, GT_global_p = motionops.R6_fk(GT_local_R6.reshape(B, T, -1, 6), GT_root_p, skeleton)

            # contact
            GT_feet_vel = GT_global_p[:, 1:, feet_ids] - GT_global_p[:, :-1, feet_ids]
            GT_feet_vel = torch.sum(torch.pow(GT_feet_vel, 2), dim=-1)
            GT_contact  = (GT_feet_vel < config.contact_vel_threshold).float()
            GT_contact  = torch.cat([GT_contact[:, 0:1], GT_contact], dim=1)

            # trajectory
            GT_root_xz    = GT_root_p[..., (0, 2)]
            GT_root_fwd   = torch.matmul(rotation.R6_to_R(GT_local_R6[..., :6]), v_forward)
            GT_root_fwd   = F.normalize(GT_root_fwd * torchconst.XZ(device), dim=-1)
            GT_root_angle = torch.atan2(GT_root_fwd[..., 0], GT_root_fwd[..., 2]) # arctan2(x, z)
            GT_traj       = torch.cat([GT_root_xz, GT_root_angle.unsqueeze(-1)], dim=-1)

            """ 2. Forward ContextVAE """
            # normalize - forward
            GT_batch = (GT_motion - motion_mean) / motion_std
            with torch.no_grad():
                pred_motion, mask = ctx_model.sample(GT_batch, GT_traj)

            """ 3. Forward DetailVAE """
            # forward - denormalize
            pred_motion, _, pred_mu, pred_logvar = det_model.forward(pred_motion, GT_traj, mask)
            pred_motion, pred_contact = torch.split(pred_motion, [D, 4], dim=-1)
            pred_motion = pred_motion * motion_std + motion_mean

            # predicted motion data
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            _, pred_global_p = motionops.R6_fk(pred_local_R6.reshape(B, T, -1, 6), pred_root_p, skeleton)
            pred_feet_v = pred_global_p[:, 1:, feet_ids] - pred_global_p[:, :-1, feet_ids]
            pred_feet_v = torch.sum(torch.pow(pred_feet_v, 2), dim=-1)
            pred_feet_v = torch.cat([pred_feet_v[:, 0:1], pred_feet_v], dim=1)

            # predicted trajectory
            pred_root_xz    = pred_root_p[..., (0, 2)]
            pred_root_fwd   = torch.matmul(rotation.R6_to_R(pred_local_R6[..., :6]), v_forward)
            pred_root_fwd   = F.normalize(pred_root_fwd * torchconst.XZ(device), dim=-1)
            pred_root_angle = torch.atan2(pred_root_fwd[..., 0], pred_root_fwd[..., 2]) # arctan2(x, z)

            """ 3. Loss """
            loss_pose    = config.weight_pose    * (trainutil.loss_recon(pred_global_p, GT_global_p) + trainutil.loss_recon(pred_local_R6, GT_local_R6))
            loss_traj    = config.weight_traj    * (trainutil.loss_recon(pred_root_xz, GT_root_xz))
            loss_contact = config.weight_contact * (trainutil.loss_recon(pred_contact, GT_contact))
            loss_foot    = config.weight_foot    * (trainutil.loss_foot(pred_contact.detach(), pred_feet_v))
            loss_kl      = config.weight_kl      * (trainutil.loss_kl(pred_mu, pred_logvar))
            loss = loss_pose + loss_traj + loss_contact + loss_foot + loss_kl

            """ 4. Backward """
            optim.zero_grad()
            loss.backward()
            optim.step()

            """ 5. Log """
            loss_dict["total"]   += loss.item()
            loss_dict["pose"]    += loss_pose.item()
            loss_dict["traj"]    += loss_traj.item()
            loss_dict["contact"] += loss_contact.item()
            loss_dict["foot"]    += loss_foot.item()
            loss_dict["kl"]      += loss_kl.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Pose: {loss_dict['pose'] / config.log_interval:.4f} | Traj: {loss_dict['traj'] / config.log_interval:.4f} | Contact: {loss_dict['contact'] / config.log_interval:.4f} | Foot: {loss_dict['foot'] / config.log_interval:.4f} | KL: {loss_dict['kl'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total",   loss_dict["total"]   / config.log_interval, iter)
                writer.add_scalar("loss/pose",    loss_dict["pose"]    / config.log_interval, iter)
                writer.add_scalar("loss/traj",    loss_dict["traj"]    / config.log_interval, iter)
                writer.add_scalar("loss/contact", loss_dict["contact"] / config.log_interval, iter)
                writer.add_scalar("loss/foot",    loss_dict["foot"]    / config.log_interval, iter)
                writer.add_scalar("loss/kl",      loss_dict["kl"]      / config.log_interval, iter)
                
                for k in loss_dict.keys():
                    loss_dict[k] = 0
            
            """ 6. Validation """
            if iter % config.val_interval == 0:
                det_model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total": 0,
                        "pose": 0,
                        "traj": 0,
                        "contact": 0,
                        "foot": 0,
                    }
                    for GT_motion in val_dataloader:
                        transition = config.max_transition
                        T = config.context_frames + transition + 1
                        GT_motion = GT_motion[:, :T, :].to(device)
                        B, T, D = GT_motion.shape
                        
                        """ 1. GT motion data """
                        # motion
                        GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
                        _, GT_global_p = motionops.R6_fk(GT_local_R6.reshape(B, T, -1, 6), GT_root_p, skeleton)

                        # contact
                        GT_feet_vel = GT_global_p[:, 1:, feet_ids] - GT_global_p[:, :-1, feet_ids]
                        GT_feet_vel = torch.sum(torch.pow(GT_feet_vel, 2), dim=-1)
                        GT_contact  = (GT_feet_vel < config.contact_vel_threshold).float()
                        GT_contact  = torch.cat([GT_contact[:, 0:1], GT_contact], dim=1)

                        # trajectory
                        GT_root_xz    = GT_root_p[..., (0, 2)]
                        GT_root_fwd   = torch.matmul(rotation.R6_to_R(GT_local_R6[..., :6]), v_forward)
                        GT_root_fwd   = F.normalize(GT_root_fwd * torchconst.XZ(device), dim=-1)
                        GT_root_angle = torch.atan2(GT_root_fwd[..., 0], GT_root_fwd[..., 2]) # arctan2(x, z)
                        GT_traj       = torch.cat([GT_root_xz, GT_root_angle.unsqueeze(-1)], dim=-1)

                        """ 2. Forward ContextVAE """
                        # normalize - forward
                        GT_batch = (GT_motion - motion_mean) / motion_std
                        pred_motion, mask = ctx_model.sample(GT_batch, GT_traj)

                        """ 3. Forward DetailVAE """
                        # forward - denormalize
                        pred_motion, _ = det_model.sample(pred_motion, GT_traj, mask)
                        pred_motion, pred_contact = torch.split(pred_motion, [D, 4], dim=-1)
                        pred_motion = pred_motion * motion_std + motion_mean

                        # predicted motion data
                        pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
                        _, pred_global_p = motionops.R6_fk(pred_local_R6.reshape(B, T, -1, 6), pred_root_p, skeleton)
                        pred_feet_v = pred_global_p[:, 1:, feet_ids] - pred_global_p[:, :-1, feet_ids]
                        pred_feet_v = torch.sum(torch.pow(pred_feet_v, 2), dim=-1)
                        pred_feet_v = torch.cat([pred_feet_v[:, 0:1], pred_feet_v], dim=1)

                        # predicted trajectory
                        pred_root_xz    = pred_root_p[..., (0, 2)]
                        pred_root_fwd   = torch.matmul(rotation.R6_to_R(pred_local_R6[..., :6]), v_forward)
                        pred_root_fwd   = F.normalize(pred_root_fwd * torchconst.XZ(device), dim=-1)
                        pred_root_angle = torch.atan2(pred_root_fwd[..., 0], pred_root_fwd[..., 2]) # arctan2(x, z)

                        """ 3. Loss """
                        loss_pose    = config.weight_pose    * (trainutil.loss_recon(pred_global_p, GT_global_p) + trainutil.loss_recon(pred_local_R6, GT_local_R6))
                        loss_traj    = config.weight_traj    * (trainutil.loss_recon(pred_root_xz, GT_root_xz))
                        loss_contact = config.weight_contact * (trainutil.loss_recon(pred_contact, GT_contact))
                        loss_foot    = config.weight_foot    * (trainutil.loss_foot(pred_contact, pred_feet_v))
                        loss = loss_pose + loss_traj + loss_contact + loss_foot

                        # log
                        val_loss_dict["total"]   += loss.item()
                        val_loss_dict["pose"]    += loss_pose.item()
                        val_loss_dict["traj"]    += loss_traj.item()
                        val_loss_dict["contact"] += loss_contact.item()
                        val_loss_dict["foot"]    += loss_foot.item()
                        
                    tqdm.write(f"Iter {iter} | Val Loss: {val_loss_dict['total'] / len(val_dataloader):.4f} | Val Pose: {val_loss_dict['pose'] / len(val_dataloader):.4f} | Val Traj: {val_loss_dict['traj'] / len(val_dataloader):.4f} | Val Contact: {val_loss_dict['contact'] / len(val_dataloader):.4f} | Val Foot: {val_loss_dict['foot'] / len(val_dataloader):.4f}")
                    writer.add_scalar("val_loss/total",   val_loss_dict["total"]   / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/pose",    val_loss_dict["pose"]    / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/traj",    val_loss_dict["traj"]    / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/contact", val_loss_dict["contact"] / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/foot",    val_loss_dict["foot"]    / len(val_dataloader), iter)

                    for k in val_loss_dict.keys():
                        val_loss_dict[k] = 0

                det_model.train()

            """ 7. Save checkpoint """
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(det_model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(det_model, optim, epoch, iter, config)