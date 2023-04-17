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
from model.gan import ContextGAN
from utility import trainutil

def get_motion_and_trajectory(motion, skeleton):
    B, T, D = motion.shape

    # motion
    local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)
    _, global_p = motionops.R6_fk(local_R6.reshape(B, T, -1, 6), root_p, skeleton)

    # trajectory
    root_xz = root_p[..., (0, 2)]
    root_fwd = torch.matmul(rotation.R6_to_R(local_R6[..., :6]), torchconst.FORWARD(motion.device))
    root_fwd = F.normalize(root_fwd * torchconst.XZ(motion.device), dim=-1)
    traj = torch.cat([root_xz, root_fwd], dim=-1)

    return local_R6.reshape(B, T, -1), global_p.reshape(B, T, -1), traj

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

    # foot joint indices
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    # model
    print("Initializing model...")
    model = ContextGAN(dataset.shape[-1], config).to(device)
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
        "gen":     0,
        "disc":    0,
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
            GT_local_R6, GT_global_p, GT_traj = get_motion_and_trajectory(GT_motion, skeleton)

            """ 2. Train Discriminator """
            # normalize - generate
            GT_motion = (GT_motion - motion_mean) / motion_std
            p_unmask = max(0, (50 - (epoch-1)//2) / 100)
            pred_context, _ = model.generate(GT_motion, GT_traj, p_unmask=p_unmask)

            # discriminator
            disc_fake_short, disc_fake_long = model.discriminate(pred_context.detach())
            disc_real_short, disc_real_long = model.discriminate(GT_motion)

            # loss
            loss_disc_short = config.weight_adv * (trainutil.loss_disc(disc_fake_short, disc_real_short))
            loss_disc_long  = config.weight_adv * (trainutil.loss_disc(disc_fake_long, disc_real_long))
            loss_disc = loss_disc_short + loss_disc_long

            # update
            optim.zero_grad()
            loss_disc.backward()
            optim.step()

            # log
            loss_dict["total"] += loss_disc.item()
            loss_dict["disc"] += loss_disc.item()

            """ 3. Train Generator """
            # generate - denormalize
            pred_motion, _ = model.generate(GT_motion, GT_traj, p_unmask=p_unmask)

            # discriminator
            disc_fake_short, disc_fake_long = model.discriminate(pred_motion)

            # predicted motion and trajectory
            pred_motion = pred_motion * motion_std + motion_mean
            pred_local_R6, pred_global_p, pred_traj = get_motion_and_trajectory(pred_motion, skeleton)

            # loss
            loss_pose    = config.weight_pose    * (trainutil.loss_recon(pred_global_p, GT_global_p, config) + trainutil.loss_recon(pred_local_R6, GT_local_R6, config))
            loss_traj    = config.weight_traj    * (trainutil.loss_traj(pred_traj, GT_traj, config))
            loss_smooth  = config.weight_smooth  * (trainutil.loss_smooth(pred_global_p) + trainutil.loss_smooth(pred_local_R6))
            loss_gen     = config.weight_adv     * (trainutil.loss_gen(disc_fake_short) + trainutil.loss_gen(disc_fake_long))
            loss         = loss_pose + loss_traj + loss_smooth + loss_gen

            """ 4. Backward """
            optim.zero_grad()
            loss.backward()
            optim.step()

            """ 5. Log """
            loss_dict["total"]   += loss.item()
            loss_dict["pose"]    += loss_pose.item()
            loss_dict["smooth"]  += loss_smooth.item()
            loss_dict["traj"]    += loss_traj.item()
            loss_dict["gen"]     += loss_gen.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Pose: {loss_dict['pose'] / config.log_interval:.4f} | Smooth: {loss_dict['smooth'] / config.log_interval:.4f} | Traj: {loss_dict['traj'] / config.log_interval:.4f} | Gen: {loss_dict['gen'] / config.log_interval:.4f} | Disc: {loss_dict['disc'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time)/60:.2f} min")
                writer.add_scalar("loss/total", loss_dict["total"]  / config.log_interval, iter)
                writer.add_scalar("loss/pose",  loss_dict["pose"]   / config.log_interval, iter)
                writer.add_scalar("loss/smooth",loss_dict["smooth"] / config.log_interval, iter)
                writer.add_scalar("loss/traj",  loss_dict["traj"]   / config.log_interval, iter)
                writer.add_scalar("loss/gen",   loss_dict["gen"]    / config.log_interval, iter)
                
                for k in loss_dict.keys():
                    loss_dict[k] = 0
            
            """ 6. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total":    0,
                        "pose":     0,
                        "traj":     0,
                        "smooth":   0,
                    }
                    for GT_motion in val_dataloader:
                        transition = config.max_transition
                        T = config.context_frames + transition + 1
                        GT_motion = GT_motion[:, :T, :].to(device)
                        B, T, D = GT_motion.shape
                        
                        """ 1. GT motion data """
                        # motion
                        GT_local_R6, GT_global_p, GT_traj = get_motion_and_trajectory(GT_motion, skeleton)

                        """ 2. Generate """
                        # generate
                        GT_motion = (GT_motion - motion_mean) / motion_std
                        pred_motion, _ = model.generate(GT_motion, GT_traj, p_unmask=0.0)
                        pred_motion = pred_motion * motion_std + motion_mean

                        # motion
                        pred_local_R6, pred_global_p, pred_traj = get_motion_and_trajectory(pred_motion, skeleton)

                        """ 3. Loss """
                        loss_pose    = config.weight_pose    * (trainutil.loss_recon(pred_global_p, GT_global_p, config) + trainutil.loss_recon(pred_local_R6, GT_local_R6, config))
                        loss_traj    = config.weight_traj    * (trainutil.loss_traj(pred_traj, GT_traj, config))
                        loss_smooth  = config.weight_smooth  * (trainutil.loss_smooth(pred_global_p) + trainutil.loss_smooth(pred_local_R6))
                        loss         = loss_pose + loss_traj + loss_smooth

                        val_loss_dict["total"]   += loss.item()
                        val_loss_dict["pose"]    += loss_pose.item()
                        val_loss_dict["traj"]    += loss_traj.item()
                        val_loss_dict["smooth"]  += loss_smooth.item()

                    tqdm.write(f"Iter {iter} | Val Loss: {val_loss_dict['total'] / len(val_dataloader):.4f} | Val Pose: {val_loss_dict['pose'] / len(val_dataloader):.4f} | Val Traj: {val_loss_dict['traj'] / len(val_dataloader):.4f} | Val Smooth: {val_loss_dict['smooth'] / len(val_dataloader):.4f}")
                    writer.add_scalar("val_loss/total", val_loss_dict["total"]  / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/pose",  val_loss_dict["pose"]   / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/traj",  val_loss_dict["traj"]   / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/smooth",val_loss_dict["smooth"] / len(val_dataloader), iter)

                model.train()

            """ 7. Save checkpoint """
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config)