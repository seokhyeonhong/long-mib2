import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader

from pymovis.ops import motionops, rotation, mathops

from utility.dataset import MotionDataset
from utility.config import Config
from utility import benchmark, utils

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/dataset.json")

    # dataset
    print("Loading dataset...")
    dataset   = MotionDataset(train=False, config=config)
    skeleton  = dataset.skeleton
    v_forward = torch.from_numpy(config.v_forward).to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # evaluation
    eval_dict = {
        "L2P":  [],
        "L2Q":  [],
        "NPSS": [],
        "L2T":  []
    }
    for GT_motion in dataloader:
        """ 1. GT motion """
        transition = 5
        T = config.context_frames + transition + 1
        GT_motion = GT_motion[:, :T, :-4].to(device) # exclude trajectory
        B, T, D = GT_motion.shape

        # trajectory        
        GT_traj = utils.get_trajectory(GT_motion, v_forward)

        # motion
        GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
        GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
        GT_global_R6, GT_global_p = motionops.R6_fk(GT_local_R6, GT_root_p, skeleton)
        GT_global_Q = rotation.R6_to_Q(GT_global_R6)

        """ 2. Interpolation """
        interp_motion = utils.get_interpolated_motion(GT_motion, config.context_frames)

        # trajectory
        interp_traj = utils.get_interpolated_trajectory(GT_traj, config.context_frames)

        # motion
        interp_local_R6, interp_root_p = torch.split(interp_motion, [D-3, 3], dim=-1)
        interp_local_R6 = interp_local_R6.reshape(B, T, -1, 6)
        interp_global_R6, interp_global_p = motionops.R6_fk(interp_local_R6, interp_root_p, skeleton)
        interp_global_Q = rotation.R6_to_Q(interp_global_R6)
        
        """ 3. Evaluation """
        l2p = benchmark.L2P(interp_global_p, GT_global_p, config.context_frames)
        l2q = benchmark.L2Q(interp_global_Q, GT_global_Q, config.context_frames)
        npss = benchmark.NPSS(interp_global_Q.reshape(B, T, -1), GT_global_Q.reshape(B, T, -1), config.context_frames)
        err_traj = benchmark.L2T(interp_traj, GT_traj, config.context_frames)

        eval_dict["L2P"].append(l2p)
        eval_dict["L2Q"].append(l2q)
        eval_dict["NPSS"].append(npss)
        eval_dict["L2T"].append(err_traj)

    for key in eval_dict.keys():
        eval_dict[key] = torch.cat(eval_dict[key], dim=0) if key is not "NPSS" else torch.stack(eval_dict[key], dim=0)
        print(f"{key}: {torch.mean(eval_dict[key]).item():.4f}")