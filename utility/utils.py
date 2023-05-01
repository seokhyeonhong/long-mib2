import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pymovis.ops import motionops, rotation, mathops
from pymovis.utils import torchconst

""" Training and test utility functions"""
def save_ckpt(model, optim, epoch, iter, config, scheduler=None):
    ckpt_path = os.path.join(config.save_dir, f"ckpt_{iter:08d}.pth")
    ckpt = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "iter": iter,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    
    torch.save(ckpt, ckpt_path)

def load_latest_ckpt(model, optim, config, scheduler=None):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
        
    ckpt_list = os.listdir(config.save_dir)
    ckpt_list = [f for f in ckpt_list if f.endswith(".pth")]
    ckpt_list = sorted(ckpt_list)
    if len(ckpt_list) > 0:
        ckpt_path = os.path.join(config.save_dir, ckpt_list[-1])
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        epoch = ckpt["epoch"]
        iter = ckpt["iter"]
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Checkpoint loaded: {ckpt_path}, epoch: {epoch}, iter: {iter}")
    else:
        epoch = 1
        iter = 1
        print("No checkpoint found. Start training from scratch.")

    return epoch, iter

def get_noam_scheduler(config, optim):
    warmup_iters = config.warmup_iters

    def _lr_lambda(iter, warmup_iters=warmup_iters):
        if iter < warmup_iters:
            return iter * warmup_iters ** (-1.5)
        else:
            return (iter ** (-0.5))
    
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_lr_lambda)

def write_log(writer, loss_dict, interval, iter, elapsed=None, train=True):
    if not train:
        tqdm.write("==============================================")
    msg = f"{'Train' if train else 'Val'} at {iter}: "
    for key, value in loss_dict.items():
        writer.add_scalar(f"{'train' if train else 'val'}/{key}", value / interval, iter)
        msg += f"{key}: {value / interval:.4f} | "
    if elapsed is not None:
        msg += f"Time: {(elapsed / 60):.2f} min"
    tqdm.write(msg)
    if not train:
        tqdm.write("==============================================")

def reset_log(loss_dict):
    for key in loss_dict.keys():
        loss_dict[key] = 0

def load_model(model, config, iter=None):
    ckpt_list = os.listdir(config.save_dir)
    if len(ckpt_list) > 0:
        if iter is None:
            ckpt_list = [ckpt for ckpt in ckpt_list if ckpt.endswith(".pth")]
            ckpt_list = sorted(ckpt_list)
            ckpt_path = os.path.join(config.save_dir, ckpt_list[-1])
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            ckpt_path = os.path.join(config.save_dir, f"ckpt_{iter:08d}.pth")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {ckpt_path}")
    else:
        raise Exception("No checkpoint found.")
    
""" Motion utility functions """
def get_motion_and_trajectory(motion, skeleton, v_forward):
    B, T, D = motion.shape

    # motion
    local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)
    _, global_p = motionops.R6_fk(local_R6.reshape(B, T, -1, 6), root_p, skeleton)

    # trajectory on xz plane
    root_fwd = torch.matmul(rotation.R6_to_R(local_R6[..., :6]), v_forward)
    root_fwd = F.normalize(root_fwd * torchconst.XZ(motion.device), dim=-1)
    traj = torch.cat([root_p[..., (0, 2)], root_fwd[..., (0, 2)]], dim=-1)

    return local_R6.reshape(B, T, -1, 6), global_p.reshape(B, T, -1, 3), traj

def get_motion(motion, skeleton):
    B, T, D = motion.shape

    # motion
    local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)
    _, global_p = motionops.R6_fk(local_R6.reshape(B, T, -1, 6), root_p, skeleton)

    return local_R6.reshape(B, T, -1, 6), global_p.reshape(B, T, -1, 3)

def get_trajectory(motion, v_forward):
    B, T, D = motion.shape

    # motion
    local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)

    # trajectory
    root_xz = root_p[..., (0, 2)]
    root_fwd = torch.matmul(rotation.R6_to_R(local_R6[..., :6]), v_forward)
    root_fwd = F.normalize(root_fwd * torchconst.XZ(motion.device), dim=-1)
    traj = torch.cat([root_xz, root_fwd], dim=-1)

    return traj

def get_velocity_and_contact(global_p, joint_ids, threshold):
    feet_v = global_p[:, 1:, joint_ids] - global_p[:, :-1, joint_ids]
    feet_v = torch.sum(feet_v**2, dim=-1) # squared norm
    feet_v = torch.cat([feet_v[:, 0:1], feet_v], dim=1)
    contact = (feet_v < threshold).float()
    return feet_v, contact

def get_interpolated_trajectory(traj, context_frames):
    # B, T, 5 = traj.shape
    res = traj.clone()

    traj_from = traj[:, context_frames-1].unsqueeze(1)
    traj_to   = traj[:, -1].unsqueeze(1)

    T = torch.linspace(0, 1, traj.shape[1] - context_frames + 1, device=traj.device)[None, :, None]

    # linear interpolation
    xz_from, xz_to = traj_from[..., :2], traj_to[..., :2]
    xz = xz_from + (xz_to - xz_from) * T
    res[:, context_frames-1:, :2] = xz
    
    # spherical linear interpolation
    fwd_from, fwd_to = traj_from[..., 2:], traj_to[..., 2:]
    signed_angles = mathops.signed_angle(fwd_from, fwd_to, dim=-1) * T.squeeze(-1)
    axis = torchconst.UP(traj.device)[None, None, :].repeat(signed_angles.shape[0], signed_angles.shape[1], 1)
    R = rotation.A_to_R(signed_angles, axis)
    fwd = torch.matmul(R, fwd_from.unsqueeze(-1)).squeeze(-1)
    res[:, context_frames-1:, 2:] = fwd
    
    return res

def get_interpolated_motion(motion, context_frames):
    B, T, D = motion.shape

    res = motion.clone()

    local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)
    local_R = rotation.R6_to_R(local_R6.reshape(B, T, -1, 6))

    local_R_from = local_R[:, context_frames-1].unsqueeze(1)
    local_R_to   = local_R[:, -1].unsqueeze(1)

    root_p_from = root_p[:, context_frames-1].unsqueeze(1)
    root_p_to   = root_p[:, -1].unsqueeze(1)

    t = torch.linspace(0, 1, T - context_frames + 1, device=motion.device)[None, :, None]

    # linear interpolation for root position
    root_p = root_p_from + (root_p_to - root_p_from) * t

    # spherical linear interpolation for joint orientations
    R_diff = torch.matmul(local_R_from.transpose(-1, -2), local_R_to)
    angle_diff, axis_diff = rotation.R_to_A(R_diff)
    angle_diff = t * angle_diff
    axis_diff = axis_diff.repeat(1, T - context_frames + 1, 1, 1)
    R_diff = rotation.A_to_R(angle_diff, axis_diff)
    local_R = torch.matmul(local_R_from, R_diff)

    local_R6 = rotation.R_to_R6(local_R).reshape(B, T - context_frames + 1, -1)
    
    res[:, context_frames-1:, :] = torch.cat([local_R6, root_p], dim=-1)
    return res

def get_align_Rp(motion, align_at, v_forward):
    """
    Returns:
        Aligned motion at frame `align_at`-th frame, centered at the origin and facing forward direction
    """
    B, T, D = motion.shape

    # motion
    local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)

    # difference of root position
    root_p_diff = root_p[:, align_at-1:align_at] * torchconst.XZ(motion.device)

    # difference of root orientation
    root_R   = rotation.R6_to_R(local_R6[:, align_at-1:align_at, :6])
    root_fwd = torch.matmul(root_R, v_forward)
    root_fwd = F.normalize(root_fwd * torchconst.XZ(motion.device), dim=-1)

    angle = mathops.signed_angle(root_fwd, v_forward[None, None, :], dim=-1)
    axis  = torchconst.Y(motion.device)[None, None, :].repeat(B, 1, 1)
    breakpoint()
    R_diff = rotation.A_to_R(angle, axis)

    return R_diff, root_p_diff

def align_motion(motion, R_diff, root_p_diff):
    B, T, D = motion.shape

    local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)

    # align root position
    root_p = root_p - root_p_diff

    # align root orientation
    root_R = rotation.R6_to_R(local_R6[..., :6])
    root_R = torch.matmul(R_diff, root_R)
    root_R6 = rotation.R_to_R6(root_R)

    local_R6 = torch.cat([root_R6, local_R6[..., 6:]], dim=-1)
    return torch.cat([local_R6, root_p], dim=-1)

""" Loss functions """
def kl_loss(mean, logvar):
    # mean: (B, T, D)
    # logvar: (B, T, D)
    # loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    mean = mean.reshape(mean.shape[0], -1)
    logvar = logvar.reshape(logvar.shape[0], -1)
    loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    return torch.mean(loss)

def recon_loss(pred, gt):
    return F.l1_loss(pred, gt)

def score_loss(pred, gt):
    # pred: (B, T, 1)
    # gt: (B, T, 1)
    loss = -torch.log(1 - torch.abs(pred - gt) + 1e-6)
    return torch.mean(loss)

def traj_loss(pred, gt):
    # pred: (B, T, 4)
    # gt: (B, T, 4)
    loss_xz = F.l1_loss(pred[..., :2], gt[..., :2])
    loss_fwd = F.l1_loss(1 - torch.sum(pred[..., 2:] * gt[..., 2:], dim=-1), torch.zeros_like(pred[..., 0]))
    return loss_xz + loss_fwd

def smooth_loss(pred):
    # pred: (B, T, D)
    # smoothness loss
    loss = F.l1_loss(pred[:, 1:] - pred[:, :-1], torch.zeros_like(pred[:, 1:]))
    return loss

def foot_loss(vel_foot, contact):
    return F.l1_loss(vel_foot * contact, torch.zeros_like(vel_foot))

def discriminator_loss(real, fake):
    # real: (B, T)
    # fake: (B, T)
    # discriminator loss
    loss_real = -torch.mean(torch.log(real + 1e-8))
    loss_fake = -torch.mean(torch.log(1 - fake + 1e-8))
    return loss_real + loss_fake

def generator_loss(fake):
    # fake: (B, T)
    # generator loss
    loss = -torch.mean(torch.log(fake + 1e-8))
    return loss