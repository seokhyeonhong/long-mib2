import copy
import random
import numpy as np
import glfw
import glm
from OpenGL.GL import *

from pymovis.vis import MotionApp, Render, YBOT_FBX_DICT

class SingleMotionApp(MotionApp):
    def __init__(self, motion, ybot_model, frames_per_motion):
        super().__init__(motion, ybot_model, YBOT_FBX_DICT)
        self.frames_per_motion = frames_per_motion
        self.show_transition = False
        self.show_many = False
        self.copy_model = copy.deepcopy(ybot_model)
        self.sphere = Render.sphere(0.05).set_albedo([1, 0, 0])

    def render(self):
        super().render()

        if self.show_many:
            if self.show_transition:
                ith_motion = self.frame // self.frames_per_motion
                for i in range(0, self.frames_per_motion, 3):
                    if i < 10:
                        self.model.set_pose_by_source(self.motion.poses[ith_motion*self.frames_per_motion + i])
                        Render.model(self.model).draw()
                    else:
                        self.copy_model.set_pose_by_source(self.motion.poses[ith_motion*self.frames_per_motion + i])
                        Render.model(self.copy_model).set_albedo_of(glm.vec3(0.5), 0).draw()

                self.model.set_pose_by_source(self.motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
                Render.model(self.model).draw()
            else:
                ith_motion = self.frame // self.frames_per_motion
                for i in range(0, 10, 3):
                    self.model.set_pose_by_source(self.motion.poses[ith_motion*self.frames_per_motion + i])
                    Render.model(self.model).draw()
                self.model.set_pose_by_source(self.motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
                Render.model(self.model).draw()
        else:
            self.model.set_pose_by_source(self.motion.poses[self.frame])
            Render.model(self.model).draw()
        
        ith_motion = self.frame // self.frames_per_motion
        for i in range(ith_motion*self.frames_per_motion, (ith_motion+1)*self.frames_per_motion):
            self.sphere.set_position(self.motion.poses[i].root_p[0], 0, self.motion.poses[i].root_p[2]).draw()

    def render_text(self):
        super().render_text()
        # Render.text_on_screen(f"Motion {self.frame // self.frames_per_motion} - Frame {self.frame % self.frames_per_motion}").set_position(10, 10, 0).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Z and action == glfw.PRESS:
            self.show_transition = not self.show_transition
        elif key == glfw.KEY_X and action == glfw.PRESS:
            self.show_many = not self.show_many

class TwoMotionApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, ybot_model, frames_per_motion, traj=None):
        super().__init__(GT_motion, ybot_model, YBOT_FBX_DICT)
        self.frames_per_motion = frames_per_motion

        # visibility
        self.axis.set_visible(False)
        self.text.set_visible(False)
        self.show_GT = True
        self.show_pred = True
        self.show_skeleton = False

        # motion and model
        self.GT_motion     = GT_motion
        self.GT_model      = ybot_model

        self.pred_motion   = pred_motion
        self.pred_model    = copy.deepcopy(ybot_model)
        self.pred_model.set_source_skeleton(self.motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.8)

        self.target_model  = copy.deepcopy(self.GT_model)

        # traj
        self.traj = traj
        self.traj_point = Render.sphere(0.05).set_albedo([1, 0, 0])
        self.show_traj = True
    
    def render(self):
        ith_motion = self.frame // self.frames_per_motion
        ith_frame = self.frame % self.frames_per_motion

        if self.show_GT:
            self.motion = self.GT_motion
            self.model = self.GT_model
            super().render(render_xray=self.show_skeleton)

        if self.show_pred:
            self.motion = self.pred_motion
            self.model = self.pred_model
            super().render(render_xray=self.show_skeleton)
        
        if self.show_traj and self.traj is not None:
            for t in range(self.frames_per_motion):
                pos = self.traj[t + ith_motion*self.frames_per_motion]
                self.traj_point.set_position(pos[0], 0, pos[1]).draw()

        # draw target
        self.GT_model.set_pose_by_source(self.GT_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
        Render.model(self.GT_model).set_all_color_modes(False).set_all_alphas(0.5).draw()
        # self.target_model.set_pose_by_source(self.GT_motion.poses[(ith_motion)*self.frames_per_motion])
        # Render.model(self.target_model).set_all_color_modes(False).set_all_alphas(0.5).draw()

    def render_text(self):
        super().render_text()
        # Render.text_on_screen(f"Motion {self.frame // self.frames_per_motion} - Frame {self.frame % self.frames_per_motion}").set_position(10, 10, 0).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        elif key == glfw.KEY_W and action == glfw.PRESS:
            self.show_pred = not self.show_pred
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.show_skeleton = not self.show_skeleton
        elif key == glfw.KEY_E and action == glfw.PRESS:
            self.show_traj = not self.show_traj

class KeyframeApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, ybot_model, frames_per_motion, keyframes, traj=None, contact=None):
        super().__init__(GT_motion, ybot_model, YBOT_FBX_DICT)

        # GT
        self.GT_motion = GT_motion
        self.GT_model  = copy.deepcopy(ybot_model)
        self.GT_model.set_source_skeleton(GT_motion.skeleton, YBOT_FBX_DICT)
        self.show_GT   = True

        # pred
        self.pred_motion = pred_motion
        self.pred_model  = copy.deepcopy(ybot_model)
        self.pred_model.set_source_skeleton(pred_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5)
        self.show_pred   = True

        # frames
        self.frames_per_motion = frames_per_motion
        self.keyframes = keyframes
        self.show_keyframe = True

        # contact
        left_leg_idx, left_foot_idx = pred_motion.skeleton.idx_by_name["LeftUpLeg"], pred_motion.skeleton.idx_by_name["LeftFoot"]
        right_leg_idx, right_foot_idx = pred_motion.skeleton.idx_by_name["RightUpLeg"], pred_motion.skeleton.idx_by_name["RightFoot"]
        ik_threshold = 0.8
        inertial_left, inertial_right = 0, 0
        count_left, count_right = 0, 0
        self.contact = contact
        if contact is not None:
            for idx, pose in enumerate(self.pred_motion.poses):
                if idx % self.frames_per_motion < 10 or idx % self.frames_per_motion == self.frames_per_motion-1:
                    inertial_left = False
                    inertial_right = False
                    count_left = 0
                    count_right = 0
                    continue
                
                if self.contact[idx, 0] > ik_threshold:
                    target_left = self.pred_motion.poses[idx-1].global_p[left_foot_idx]
                    pose.two_bone_ik(left_leg_idx, left_foot_idx, self.pred_motion.poses[idx-1].global_p[left_foot_idx])
                    inertial_left = True
                    count_left = 0
                elif inertial_left is True:
                    count_left += 1
                    disp = self.pred_motion.poses[idx].global_p[left_foot_idx] - target_left
                    pose.two_bone_ik(left_leg_idx, left_foot_idx, target_left + disp * (0.1 * count_left))
                    if count_left >= 10:
                        inertial_left = False
                        count_left = 0

                if self.contact[idx, 1] > ik_threshold:
                    target_right = self.pred_motion.poses[idx-1].global_p[right_foot_idx]
                    pose.two_bone_ik(right_leg_idx, right_foot_idx, self.pred_motion.poses[idx-1].global_p[right_foot_idx])
                    inertial_right = True
                    count_right = 0
                elif inertial_right is True:
                    count_right += 1
                    disp = self.pred_motion.poses[idx].global_p[right_foot_idx] - target_right
                    pose.two_bone_ik(right_leg_idx, right_foot_idx, target_right + disp * (0.1 * count_right))
                    if count_right >= 10:
                        inertial_right = False
                        count_right = 0
        
        # traj
        self.traj = traj
        self.traj_sphere = Render.sphere(0.05)
        self.show_traj = True
    
    def render(self):
        super().render(render_model=False)

        ith_motion = self.frame // self.frames_per_motion
        ith_frame = self.frame % self.frames_per_motion

        # GT
        if self.show_GT:
            self.GT_model.set_pose_by_source(self.GT_motion.poses[self.frame])
            Render.model(self.GT_model).set_all_alphas(1.0).draw()

        # pred
        if self.show_pred:
            self.pred_model.set_pose_by_source(self.pred_motion.poses[self.frame])
            Render.model(self.pred_model).set_all_alphas(1.0).draw()
        
        # keyframes
        # if self.show_keyframe:
        #     # for kf in self.keyframes:
        #     #     if ith_motion * self.frames_per_motion <= kf < (ith_motion+1) * self.frames_per_motion - 1:
        #     #         self.pred_model.set_pose_by_source(self.pred_motion.poses[kf])
        #     #         Render.model(self.pred_model).set_all_alphas(0.5).draw()
        #     self.pred_model.set_pose_by_source(self.pred_motion.poses[(ith_motion+1) * self.frames_per_motion - 1])
        #     Render.model(self.pred_model).set_all_alphas(0.5).draw()
        #     self.GT_model.set_pose_by_source(self.GT_motion.poses[(ith_motion+1) * self.frames_per_motion - 1])
        #     Render.model(self.GT_model).set_all_alphas(0.5).draw()

        # target frame
        self.GT_model.set_pose_by_source(self.GT_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
        Render.model(self.GT_model).set_all_alphas(0.5).draw()

        self.pred_model.set_pose_by_source(self.pred_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
        Render.model(self.pred_model).set_all_alphas(0.5).draw()

        # traj
        if self.show_traj and self.traj is not None:
            traj = self.traj[(ith_motion)*self.frames_per_motion:(ith_motion+1)*self.frames_per_motion]
            for i in range(len(traj)):
                if i != ith_frame:
                    self.traj_sphere.set_albedo([1, 0, 0]).set_position(traj[i, 0], 0, traj[i, 1]).draw()
                else:
                    self.traj_sphere.set_albedo([0, 1, 0]).set_position(traj[i, 0], 0, traj[i, 1]).draw()

    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        elif key == glfw.KEY_W and action == glfw.PRESS:
            self.show_pred = not self.show_pred
        elif key == glfw.KEY_E and action == glfw.PRESS:
            self.show_keyframe = not self.show_keyframe
        elif key == glfw.KEY_R and action == glfw.PRESS:
            self.show_traj = not self.show_traj

class TripletMotionApp(MotionApp):
    def __init__(self, GT_motion, motion1, motion2, ybot_model, frames_per_motion, traj=None, l2ps=None):
        super().__init__(GT_motion, ybot_model, YBOT_FBX_DICT)
        self.frames_per_motion = frames_per_motion

        # visibility
        self.axis.set_visible(False)
        self.text.set_visible(False)
        self.show_GT = True
        self.show_motion1 = True
        self.show_motion2 = True
        self.show_skeleton = False
        self.show_constrained = False

        # motion and model
        self.GT_motion     = GT_motion
        self.GT_model      = ybot_model
        self.GT_model.set_source_skeleton(GT_motion.skeleton, YBOT_FBX_DICT)

        # pred
        self.motion1 = motion1
        self.model1  = copy.deepcopy(ybot_model)
        self.model1.set_source_skeleton(motion1.skeleton, YBOT_FBX_DICT)
        self.model1.meshes[0].materials[0].albedo = glm.vec3(0.8)
        self.model1.meshes[1].materials[0].albedo = glm.vec3(0.2)

        self.motion2 = motion2
        self.model2  = copy.deepcopy(ybot_model)
        self.model2.set_source_skeleton(motion2.skeleton, YBOT_FBX_DICT)
        self.model2.meshes[0].materials[0].albedo = glm.vec3(0.8, 0.1, 0.1)
        self.model2.meshes[1].materials[0].albedo = glm.vec3(0.3, 0.1, 0.1)

        # move pred motions
        for pose in self.motion1.poses:
            pose.translate_root_p(np.array([1.5, 0, 0]))
        for pose in self.motion2.poses:
            pose.translate_root_p(np.array([3, 0, 0]))
        
        # target
        self.target_model  = copy.deepcopy(self.GT_model)

        # traj
        self.traj = traj
        self.traj_point = Render.sphere(0.05).set_albedo([1, 0, 0])
        self.show_traj = True

        # L2P
        self.ref_l2p = l2ps[0]
        self.det_l2p = l2ps[1]

        # guide text
        self.show_guide = True
    
        # OPTIONAL: Which model to show at center
        self.motion = self.motion1
        self.model  = self.model1
    def render_char(self, model, pose, alpha=1.0):
        model.set_pose_by_source(pose)
        Render.model(model).set_all_alphas(alpha).draw()

    def render(self):
        super().render(render_model=False, render_xray=False)
        ith_motion = self.frame // self.frames_per_motion
        ith_frame = self.frame % self.frames_per_motion

        if self.show_GT:
            self.render_char(self.GT_model, self.GT_motion.poses[self.frame], 0.5 if ith_frame == 0 or ith_frame == self.frames_per_motion - 1 else 1.0)
            if self.show_constrained:
            #     self.render_char(self.GT_model, self.GT_motion.poses[ith_motion * self.frames_per_motion], 0.5)
                self.render_char(self.GT_model, self.GT_motion.poses[(ith_motion+1) * self.frames_per_motion - 1], 0.5)

        if self.show_motion1:
            self.render_char(self.model1, self.motion1.poses[self.frame], 0.5 if ith_frame == 0 or ith_frame == self.frames_per_motion - 1 else 1.0)
            if self.show_constrained:
            #     self.render_char(self.model1, self.motion1.poses[ith_motion * self.frames_per_motion], 0.5)
                self.render_char(self.model1, self.motion1.poses[(ith_motion+1) * self.frames_per_motion - 1], 0.5)
        
        if self.show_motion2:
            self.render_char(self.model2, self.motion2.poses[self.frame], 0.5 if ith_frame == 0 or ith_frame == self.frames_per_motion - 1 else 1.0)
            if self.show_constrained:
                # self.render_char(self.model2, self.motion2.poses[ith_motion * self.frames_per_motion], 0.5)
                self.render_char(self.model2, self.motion2.poses[(ith_motion+1) * self.frames_per_motion - 1], 0.5)
        
        if self.show_traj and self.traj is not None:
            for t in range(self.frames_per_motion):
                pos = self.traj[t + ith_motion*self.frames_per_motion]
                if t != ith_frame:
                    self.traj_point.set_position(pos[0], 0, pos[1]).set_albedo([1, 0, 0]).draw()
                else:
                    self.traj_point.set_position(pos[0], 0, pos[1]).set_albedo([0, 1, 0]).draw()

        # draw target
        # self.target_model.set_pose_by_source(self.GT_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
        # Render.model(self.target_model).set_all_color_modes(False).set_all_alphas(0.5).draw()
        # self.target_model.set_pose_by_source(self.GT_motion.poses[(ith_motion)*self.frames_per_motion])
        # Render.model(self.target_model).set_all_color_modes(False).set_all_alphas(0.5).draw()

    def render_text(self):
        super().render_text()
        ith_motion = self.frame // self.frames_per_motion
        # if self.show_GT:
        #     pos = self.GT_motion.poses[self.frame].root_p
        #     Render.text("GT").set_position(pos[0], pos[1] + 0.5, pos[2]).set_scale(0.5).draw()
        # if self.show_motion1:
        #     pos = self.motion1.poses[self.frame].root_p
        #     Render.text(f"Ours - {self.ref_l2p[ith_motion].item():.4f}").set_position(pos[0], pos[1] + 0.5, pos[2]).set_scale(0.5).draw()
        # if self.show_motion2:
        #     pos = self.motion2.poses[self.frame].root_p
        #     Render.text(f"TS-Trans: {self.det_l2p[ith_motion].item():.4f}").set_position(pos[0], pos[1] + 0.5, pos[2]).set_scale(0.5).draw()
        if self.show_guide:
            Render.text_on_screen(f"Motion {self.frame // self.frames_per_motion} - Frame {self.frame % self.frames_per_motion}").set_position(10, 10, 0).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        elif key == glfw.KEY_W and action == glfw.PRESS:
            self.show_motion1 = not self.show_motion1
        elif key == glfw.KEY_E and action == glfw.PRESS:
            self.show_motion2 = not self.show_motion2
        elif key == glfw.KEY_R and action == glfw.PRESS:
            self.show_traj = not self.show_traj
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.show_skeleton = not self.show_skeleton
        elif key == glfw.KEY_D and action == glfw.PRESS:
            self.show_constrained = not self.show_constrained
        elif key == glfw.KEY_Z and action == glfw.PRESS:
            self.show_guide = not self.show_guide

class NeighborApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, neighbor_motions, ybot_model, frames_per_motion, text=None):
        super().__init__(pred_motion, ybot_model, YBOT_FBX_DICT)

        self.print_text = text

        # GT
        self.GT_motion = GT_motion
        self.GT_model  = copy.deepcopy(ybot_model)
        self.GT_model.set_source_skeleton(GT_motion.skeleton, YBOT_FBX_DICT)
        self.show_GT   = True

        for pose in self.GT_motion.poses:
            pose.translate_root_p(np.array([-1.5, 0, 0]))

        # pred
        self.pred_motion = pred_motion
        self.pred_model  = copy.deepcopy(ybot_model)
        self.pred_model.set_source_skeleton(pred_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5)
        self.pred_traj = Render.sphere(0.05).set_albedo(0.5)
        self.show_pred = True
        self.show_target = True

        # frames
        self.frames_per_motion = frames_per_motion

        def f():
            x=random.random()
            from scipy.interpolate import CubicSpline
            data_points = [0, 0.5, 1]
            values = [1, 0, 1]
            # values = [0, 1, 0]
            cs = CubicSpline(data_points, values)
            return cs(x)
    
        # neighbors
        colors = [
            glm.vec3(0.80, 0.20, 0.26),
            glm.vec3(0.02, 0.78, 0.20),
            glm.vec3(0.04, 0.17, 0.97),
            glm.vec3(0.21, 0.10, 0.25),
            glm.vec3(0.71, 0.67, 0.05),
        ]
        self.neighbor_motions = neighbor_motions
        self.show_neighbors   = True
        self.neighbor_models = []
        for idx, motion in enumerate(self.neighbor_motions):
            model = copy.deepcopy(ybot_model)
            model.set_source_skeleton(motion.skeleton, YBOT_FBX_DICT)
            model.meshes[0].materials[0].albedo = colors[idx]
            model.meshes[1].materials[0].albedo = colors[idx] * 0.5
            # model.meshes[0].materials[0].albedo = glm.vec3(f(), f(), f())
            self.neighbor_models.append(model)

            for pose in motion.poses:
                pose.translate_root_p(np.array([-1.5 * (idx+1), 0, 0]))

        # trajectory
        self.show_traj = True
        self.traj_points = []
        for i in range(len(neighbor_motions)):
            self.traj_points.append(Render.sphere(0.05).set_albedo(self.neighbor_models[i].meshes[0].materials[0].albedo))

        print("Q: GT, W: pred, E: neighbors, R: traj, S: target")

    def render(self):
        super().render(render_model=False)

        ith_motion = self.frame // self.frames_per_motion
        ith_frame = self.frame % self.frames_per_motion

        # GT
        if self.show_GT:
            self.GT_model.set_pose_by_source(self.GT_motion.poses[self.frame])
            Render.model(self.GT_model).set_all_alphas(1.0).draw()

        # pred
        if self.show_pred:
            self.pred_model.set_pose_by_source(self.pred_motion.poses[self.frame])
            Render.model(self.pred_model).set_all_alphas(1.0).draw()

            if self.show_target:
                self.pred_model.set_pose_by_source(self.pred_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
                Render.model(self.pred_model).set_all_alphas(0.5).draw()
            # Render.model(self.pred_model).set_all_alphas(1.0 if (ith_frame != 0 and ith_frame != self.frames_per_motion - 1) else 0.5).draw()
            if self.show_traj:
                for pose in self.pred_motion.poses[ith_motion*self.frames_per_motion:(ith_motion+1)*self.frames_per_motion]:
                    position = pose.root_p
                    self.pred_traj.set_position(position[0], 0, position[2]).draw()
        
        # neighbors
        if self.show_neighbors:
            for i, model in enumerate(self.neighbor_models):
                model.set_pose_by_source(self.neighbor_motions[i].poses[self.frame])
                Render.model(model).set_all_alphas(1.0).draw()

                if self.show_target:
                    model.set_pose_by_source(self.neighbor_motions[i].poses[(ith_motion+1)*self.frames_per_motion - 1])
                    Render.model(model).set_all_alphas(0.5).draw()

                if self.show_traj:
                    for pose in self.neighbor_motions[i].poses[ith_motion*self.frames_per_motion:(ith_motion+1)*self.frames_per_motion]:
                        position = pose.root_p
                        self.traj_points[i].set_position(position[0], 0, position[2]).draw()
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        elif key == glfw.KEY_W and action == glfw.PRESS:
            self.show_pred = not self.show_pred
        elif key == glfw.KEY_E and action == glfw.PRESS:
            self.show_neighbors = not self.show_neighbors
        elif key == glfw.KEY_R and action == glfw.PRESS:
            self.show_traj = not self.show_traj
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.show_target = not self.show_target
        
    def render_text(self):
        ith_motion = self.frame // self.frames_per_motion
        super().render_text()
        if self.print_text is not None:
            Render.text_on_screen(self.print_text[ith_motion]).draw()