import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import glm
import glfw
import copy
from pymovis.utils import torchconst
from pymovis.motion import Motion, FBX
from pymovis.vis import AppManager, MotionApp, YBOT_FBX_DICT, Render
from pymovis.ops import rotation

from utility.config import Config
from utility.dataset import MotionDataset
from vis.visapp import SingleMotionApp

class DatasetApp(MotionApp):
    def __init__(self, motion, model, traj):
        super().__init__(motion, model, YBOT_FBX_DICT)
        self.copy_model = copy.deepcopy(model)
        self.traj = traj

        # vis
        self.arrow = Render.arrow().set_albedo([1, 0, 0])
        self.sphere = Render.sphere(0.05, 4, 4).set_albedo([1, 0, 0])
    
    def render(self):
        super().render()

        # render trajectory
        xz, forward = self.traj[self.frame, :2], self.traj[self.frame, 2:]
        R = glm.angleAxis(glm.radians(90), glm.cross(glm.vec3(0, 1, 0), glm.vec3(forward[0], 0, forward[1])))
        p = glm.vec3(xz[0], 0, xz[1])
        self.arrow.set_position(p).set_orientation(R).draw()

        for i in range(self.prob_idx):
            self.copy_model.set_pose_by_source(self.motion.poses[self.prob_sorted_idx[i]])
            Render.model(self.copy_model).set_all_alphas(0.5).draw()
    
    def render_text(self):
        super().render_text()
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/mpvae.json")

    dataset = MotionDataset(train=False, config=config)
    v_forward = torch.from_numpy(config.v_forward).to(device)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    skeleton = dataset.skeleton

    character = FBX("dataset/ybot.fbx")

    for GT_motion in dataloader:
        """ 1. GT motion data """
        B, T, D = GT_motion.shape
        GT_motion = GT_motion.to(device)
        GT_local_R6, GT_root_p, GT_traj = torch.split(GT_motion, [D-7, 3, 4], dim=-1)
        GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

        # print(F.normalize(torch.matmul(GT_local_R[:, :, 0], v_forward) * torchconst.XZ(device), dim=-1))
        # breakpoint()

        """ 2. Animation """
        motion = Motion.from_torch(skeleton, GT_local_R.reshape(B*T, -1, 3, 3), GT_root_p.reshape(B*T, 3))

        """ 3. Visualization """
        app_manager = AppManager()
        app = DatasetApp(motion, character.model(), GT_traj.reshape(B*T, 4))
        app_manager.run(app)