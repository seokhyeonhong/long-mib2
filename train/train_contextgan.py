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
from model.ours import ContextGAN
from utility import trainutil

if __name__ == "__main__":
    # initial settings with all possible gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context_gan.json")
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

    # model
    print("Initializing model...")
    model = ContextGAN(dataset.shape[-1], config).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    init_epoch, iter = trainutil.load_latest_ckpt(model, optim, config)
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
        "smooth":  0,
        "traj":    0,
        "disc":    0,
        "gen":     0,
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

            # trajectory
            GT_root_xz    = GT_root_p[..., (0, 2)]
            GT_root_fwd   = torch.matmul(rotation.R6_to_R(GT_local_R6[..., :6]), v_forward)
            GT_root_fwd   = F.normalize(GT_root_fwd * torchconst.XZ(device), dim=-1)
            GT_root_angle = torch.atan2(GT_root_fwd[..., 0], GT_root_fwd[..., 2]) # arctan2(x, z)
            GT_traj       = torch.cat([GT_root_xz, GT_root_angle.unsqueeze(-1)], dim=-1)

            # batch
            GT_batch = (GT_motion - motion_mean) / motion_std

            """ 2. Train Discriminator """
            # generate
            fake_motion = model.generate(GT_batch, GT_traj)

            # discriminate
            disc_real_short, disc_real_long = model.discriminate(GT_batch)
            disc_fake_short, disc_fake_long = model.discriminate(fake_motion.detach())
            loss_disc = config.weight_adv * (trainutil.loss_disc(disc_fake_short, disc_real_short)\
                                             + trainutil.loss_disc(disc_fake_long, disc_real_long))
            
            # update discriminator
            optim.zero_grad()
            loss_disc.backward()
            optim.step()

            """ 3. Train Generator """
            # generate
            fake_motion = model.generate(GT_batch, GT_traj)

            # discriminate
            disc_fake_short, disc_fake_long = model.discriminate(fake_motion)
            loss_gen = config.weight_adv * (trainutil.loss_gen(disc_fake_short) + trainutil.loss_gen(disc_fake_long))

            # predicted motion
            pred_motion = pred_motion * motion_std + motion_mean
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            _, pred_global_p = motionops.R6_fk(pred_local_R6.reshape(B, T, -1, 6), pred_root_p, skeleton)

            # predicted trajectory
            pred_root_xz    = pred_root_p[..., (0, 2)]
            pred_root_fwd   = torch.matmul(rotation.R6_to_R(pred_local_R6[..., :6]), v_forward)
            pred_root_fwd   = F.normalize(pred_root_fwd * torchconst.XZ(device), dim=-1)
            pred_root_angle = torch.atan2(pred_root_fwd[..., 0], pred_root_fwd[..., 2]) # arctan2(x, z)

            # loss
            loss_pose = config.weight_pose * (trainutil.loss_recon(pred_global_p, GT_global_p)\
                                              + trainutil.loss_recon(pred_local_R6, GT_local_R6))
            loss_smooth = config.weight_smooth * (trainutil.loss_smooth(pred_global_p)\
                                                  + trainutil.loss_smooth(pred_local_R6))
            loss_traj = config.weight_traj * (trainutil.loss_recon(pred_root_xz, GT_root_xz))
            loss = loss_pose + loss_smooth + loss_traj + loss_gen

            # update generator
            optim.zero_grad()
            loss.backward()
            optim.step()

            """ 4. Log """
            loss_dict["total"]  += loss.item()
            loss_dict["pose"]   += loss_pose.item()
            loss_dict["smooth"] += loss_smooth.item()
            loss_dict["traj"]   += loss_traj.item()
            loss_dict["disc"]   += loss_disc.item()
            loss_dict["gen"]    += loss_gen.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Pose: {loss_dict['pose'] / config.log_interval:.4f} | Smooth: {loss_dict['smooth'] / config.log_interval:.4f} | Traj: {loss_dict['traj'] / config.log_interval:.4f} | Disc: {loss_dict['disc'] / config.log_interval:.4f} | Gen: {loss_dict['gen'] / config.log_interval:.4f} | Time: {(time.time() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total", loss_dict["total"]  / config.log_interval, iter)
                writer.add_scalar("loss/pose",  loss_dict["pose"]   / config.log_interval, iter)
                writer.add_scalar("loss/smooth",loss_dict["smooth"] / config.log_interval, iter)
                writer.add_scalar("loss/traj",  loss_dict["traj"]   / config.log_interval, iter)
                writer.add_scalar("loss/disc",  loss_dict["disc"]   / config.log_interval, iter)
                writer.add_scalar("loss/gen",   loss_dict["gen"]    / config.log_interval, iter)
                
                for k in loss_dict.keys():
                    loss_dict[k] = 0
            
            """ 5. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total": 0,
                        "pose": 0,
                        "smooth": 0,
                        "traj": 0,
                        "kl": 0,
                    }
                    for GT_motion in tqdm(val_dataloader, desc="Validation"):
                        transition = config.max_transition
                        T = config.context_frames + transition + 1
                        GT_motion = GT_motion[:, :T, :].to(device)
                        B, T, D = GT_motion.shape
                        
                        """ 1. GT motion data """
                        # motion
                        GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
                        _, GT_global_p = motionops.R6_fk(GT_local_R6.reshape(B, T, -1, 6), GT_root_p, skeleton)

                        # trajectory
                        GT_root_xz    = GT_root_p[..., (0, 2)]
                        GT_root_fwd   = torch.matmul(rotation.R6_to_R(GT_local_R6[..., :6]), v_forward)
                        GT_root_fwd   = F.normalize(GT_root_fwd * torchconst.XZ(device), dim=-1)
                        GT_root_angle = torch.atan2(GT_root_fwd[..., 0], GT_root_fwd[..., 2]) # arctan2(x, z)
                        GT_traj       = torch.cat([GT_root_xz, GT_root_angle.unsqueeze(-1)], dim=-1)

                        """ 2. Forward ContextVAE """
                        # normalize - forward - denormalize
                        GT_batch = (GT_motion - motion_mean) / motion_std
                        pred_motion, pred_mu, pred_logvar = model(GT_batch, GT_traj)
                        pred_motion = pred_motion * motion_std + motion_mean

                        # predicted motion data
                        pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
                        _, pred_global_p = motionops.R6_fk(pred_local_R6.reshape(B, T, -1, 6), pred_root_p, skeleton)

                        # predicted trajectory
                        pred_root_xz    = pred_root_p[..., (0, 2)]
                        pred_root_fwd   = torch.matmul(rotation.R6_to_R(pred_local_R6[..., :6]), v_forward)
                        pred_root_fwd   = F.normalize(pred_root_fwd * torchconst.XZ(device), dim=-1)
                        pred_root_angle = torch.atan2(pred_root_fwd[..., 0], pred_root_fwd[..., 2]) # arctan2(x, z)

                        """ 3. Loss """
                        loss_pose = config.weight_pose * (trainutil.loss_recon(pred_global_p, GT_global_p)\
                                                        + trainutil.loss_recon(pred_local_R6, GT_local_R6))
                        loss_smooth = config.weight_smooth * (trainutil.loss_smooth(pred_global_p)\
                                                            + trainutil.loss_smooth(pred_local_R6))
                        loss_traj = config.weight_traj * (trainutil.loss_recon(pred_root_xz, GT_root_xz))
                        loss_kl   = config.weight_kl * trainutil.loss_kl(pred_mu, pred_logvar)
                        loss = loss_pose + loss_smooth + loss_traj + loss_kl

                        # log
                        val_loss_dict["total"]   += loss.item()
                        val_loss_dict["pose"]    += loss_pose.item()
                        val_loss_dict["smooth"]  += loss_smooth.item()
                        val_loss_dict["traj"]    += loss_traj.item()
                        val_loss_dict["kl"]      += loss_kl.item()

                    tqdm.write(f"Iter {iter} | Val Loss: {val_loss_dict['total'] / len(val_dataloader):.4f} | Val Pose: {val_loss_dict['pose'] / len(val_dataloader):.4f} | Val Traj: {val_loss_dict['traj'] / len(val_dataloader):.4f}")
                    writer.add_scalar("val_loss/total", val_loss_dict["total"]  / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/pose",  val_loss_dict["pose"]   / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/smooth",val_loss_dict["smooth"] / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/traj",  val_loss_dict["traj"]   / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/kl",    val_loss_dict["kl"]     / len(val_dataloader), iter)

                model.train()

            """ 7. Save checkpoint """
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config)