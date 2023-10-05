# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import math
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

import cv2
import imageio

from collections import defaultdict 



class CartpoleSample(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # [CH] train.py:: envs = isaacgymenvs.make(...)
        self.cfg = cfg
        
        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 2400

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 1

        # [CH] added configs via command, eg, +task.env.dumped_imgs_dir='./output/CartpoleSample/dumped_imgs' 
        self.is_train           = self.cfg["env"].get("is_train", True)
        self.use_cam            = self.cfg["env"].get("use_cam", False)
        self.debug              = self.cfg["env"].get("debug", False)
        self.sample             = self.cfg["env"].get("sample", False)
        self.camera_resolution  = self.cfg["env"].get("camera_resolution", 256)
        self.max_frame_length   = self.cfg["env"].get("max_frame_length", 900)  
        self.seed               = self.cfg["env"].get("seed", 42)

        dump_imgs_dir = "./output/CartpoleSample/{}/".format("debug" if self.debug else "inference")
        self.dumped_imgs_dir    = self.cfg["env"].get("dumped_imgs_dir", dump_imgs_dir) 
        
        # [CH] create path for dumped images
        if not os.path.exists(self.dumped_imgs_dir):
            os.makedirs(self.dumped_imgs_dir)

        # [CH] set camera
        self.camera_width = self.camera_resolution
        self.camera_height = self.camera_resolution

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Initialization of observations 
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # [CH] Randomize init dof_state
        self.reset_idx(torch.arange(self.num_envs).to(self.device))
        self.gym.refresh_dof_state_tensor(self.sim)

        self.gym.simulate(self.sim)
        self.frame_no = 0
        
        if self.debug:
            print("Debug mode set... ")
            print(self.dof_state)
            self.render_from_cam(self.debug)
            exit(0)

        # [CH] sample data
        if self.sample:
            print("Sample mode set... ")
            sequence_dir = "./data"
            if not os.path.exists(sequence_dir):
                os.makedirs(sequence_dir)
            self.sequence_path = os.path.join(sequence_dir,"seq_{}-{}-{}-{}.pt".format(self.cfg["name"], self.seed, self.num_environments, self.max_frame_length))
            self.sequence = defaultdict()
            self.sequence["x"], self.sequence["theta"] = self.get_x_theta(self.compute_observations())
            

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        # [CH] called in super().__init__(...)
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def get_x_theta(self, obs):
        pos = obs.detach().clone().cpu()[:,[0,2]]
        return pos[:,[0]], pos[:,[1]] # cart_pos, angle_pos

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.cartpole_handles = []
        self.envs = []

        # [CH] Add camera
        self.cams = []
        self.cam_tensors = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

            if self.use_cam:
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.camera_width
                cam_props.height = self.camera_height
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)  

                cam_pos = gymapi.Vec3(1.3, 0.0, 2.5) if self.up_axis=='z' else gymapi.Vec3(1.3, 2.5, 0.0)
                cam_target = gymapi.Vec3(0, 0, 2.3) if self.up_axis=='z' else gymapi.Vec3(0, 2.3, 0)

                self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)
                self.cams.append(cam_handle)

                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(torch_cam_tensor)


    def compute_reward(self):
        # retrieve environment observations from buffer
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]
        cart_vel = self.obs_buf[:, 1]
        cart_pos = self.obs_buf[:, 0]

        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.9 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        # velocities = 0.5 * (torch.ones((len(env_ids), self.num_dof), device=self.device))
        # positions = 0.9 * (torch.ones((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    # [CH] actions <- get_action(): https://github.com/Denys88/rl_games/blob/66ce12f30f2582d43c818356baba1812669841db/rl_games/algos_torch/players.py#L45
    def pre_physics_step(self, actions):
        # [CH] The VecTask class is itself the env. See comments in vec_task.py for details.
        # [CH] Player has env. Player calls obs=env.reset(). It DOES NOT get current but RESET obs. See VecTask::reset().
        # [CH] To get observations, do obs=compute_observations().
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    # [CH] camera screenshot
    def render_from_cam(self, is_debug=False):
        #self.gym.render_all_camera_sensors(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            if not is_debug and ((i%self.num_envs!=0) and (i%self.num_envs!=1) and (i%self.num_envs!=2) and (i%self.num_envs!=3)):
                continue

            fname = os.path.join(self.dumped_imgs_dir, "cam-%04d-%04d.png" % (i, self.frame_no))
            cam_img = self.cam_tensors[i].cpu().numpy()
            imageio.imwrite(fname, cam_img)
        self.gym.draw_viewer(self.viewer, self.sim, True)


    def post_physics_step(self):

        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        obs = self.compute_observations()
        self.compute_reward()

        if self.sample or not self.is_train:
            if self.frame_no >= self.max_frame_length:
                if self.sample:
                    print("Done...")
                    torch.save(self.sequence,self.sequence_path)
                exit(0)

        # [CH] sample 
        if self.sample:
            x, theta = self.get_x_theta(obs)
            self.sequence["x"] = torch.cat((self.sequence["x"],x),dim=1)
            self.sequence["theta"] = torch.cat((self.sequence["theta"],theta),dim=1)
            

        # [CH] render images
        if not self.is_train:
            self.render_from_cam()
            self.frame_no += 1
        

    # [CH] Write data every step
    # def get_sampled_obs_action_pair(self):
    #     '''[CH]
    #     Call this function once done sampling
    #     '''
    #     return self.sampled_obs_action_pair
    


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_cartpole_reward(pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - 0.001 * pole_angle * pole_angle - 0.001 * torch.abs(cart_vel) - 0.001 * torch.abs(pole_vel)
    # reward = 1.0 - pole_angle * pole_angle - 0.01 * cart_vel * cart_vel - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
    #reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -100.0, reward)
    #reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -100.0, reward)

    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset
