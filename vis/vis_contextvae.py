import sys
sys.path.append(".")

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import glm, glfw
from tqdm import tqdm

from pymovis.utils import util, torchconst
from pymovis.motion import Motion, FBX
from pymovis.ops import rotation
from pymovis.vis import AppManager, MotionApp, Render, YBOT_FBX_DICT

from utility import testutil
from utility.config import Config
from utility.dataset import MotionDataset
from model.vae import VAE

class KeyframeApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, model, time_per_motion):
        super().__init__(GT_motion, model, YBOT_FBX_DICT)

        self.GT_motion = GT_motion
        self.pred_motion = pred_motion

        self.show_GT = True
        self.show_pred = True

        self.time_per_motion = time_per_motion

        self.GT_model = model
        self.GT_model.set_source_skeleton(self.GT_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model = copy.deepcopy(model)
        self.pred_model.set_source_skeleton(self.pred_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

    def render(self):
        super().render(render_model=False)

        if self.show_GT:
            self.GT_model.set_pose_by_source(self.GT_motion.poses[self.frame])
            Render.model(self.GT_model).draw()
            # Render.model(self.GT_model).set_all_alphas(self.GT_prob[self.frame]).draw()
        if self.show_pred:
            self.pred_model.set_pose_by_source(self.pred_motion.poses[self.frame])
            Render.model(self.pred_model).draw()

    def render_text(self):
        super().render_text()
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        if key == glfw.KEY_W and action == glfw.PRESS:
            self.show_pred = not self.show_pred

def get_mask(batch, context_frames):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones_like(batch)
    batch_mask[:, context_frames:-1, :] = 0

    # TODO: Add probability of unmasking

    return batch_mask

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context_vae.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    # mean and std
    motion_mean, motion_std = dataset.statistics()
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = VAE(dataset.shape[-1], config, is_context=True).to(device)
    testutil.load_model(model, config)
    model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T, :].to(device)
            B, T, D = GT_motion.shape

            """ 1. Prepare GT data """
            # motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # trajectory
            GT_root_xz  = GT_root_p[..., (0, 2)]
            GT_root_fwd = torch.matmul(rotation.R6_to_R(GT_local_R6[..., :6]), v_forward)
            GT_root_fwd = F.normalize(GT_root_fwd * torchconst.XZ(device), dim=-1)
            GT_traj     = torch.cat([GT_root_xz, GT_root_fwd], dim=-1)

            """ 1-1. Modify Trajectory """
            # traj_from = GT_traj[:, config.context_frames-1, -3:].unsqueeze(1)
            # traj_to   = GT_traj[:, -1, -3:].unsqueeze(1)
            # t = torch.linspace(0, 1, T - config.context_frames + 1)[:, None].to(device)
            # GT_traj[:, config.context_frames-1:, -3:] = traj_from + (traj_to - traj_from) * t
            # GT_traj[:, config.context_frames:, -3] = torch.linspace(0, torch.pi, T - config.context_frames)[None, :].to(device)
            # GT_traj[:, config.context_frames:, -2] = torch.sin(GT_traj[:, config.context_frames:, -3])
            # GT_traj[:, config.context_frames:, -1] = GT_traj[:, config.context_frames-1, -1].unsqueeze(1).clone()

            # GT_root_p[:, -1, (0, 2)] = GT_traj[:, -1, (-3, -2)]
            # GT_local_R[:, -1, 0] = GT_local_R[:, config.context_frames-1, 0]

            """ 2. Sample """
            # normalize - forward - denormalize
            GT_batch = (GT_motion - motion_mean) / motion_std
            mask = get_mask(GT_batch, config.context_frames)
            pred_motion = model.sample(GT_batch, GT_traj, mask)
            pred_motion = pred_motion * motion_std + motion_mean

            # predicted motion data
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))
            
            """ 3. Visualize """
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            app = KeyframeApp(GT_motion, pred_motion, ybot.model(), T)
            app_manager.run(app)