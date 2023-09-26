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
import os
import math, sys, imageio
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import compute_heading_and_up, compute_rot, \
    unscale, to_torch, get_axis_params, quat_conjugate, torch_rand_float, \
    tensor_clamp
from isaacgymenvs.tasks.base.vec_task import VecTask

import cv2


class AntImitate(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        # visualization logic, added by Nate
        self.num_envs_to_visualize = 1
        self.num_frames_to_visualize = math.inf
        self.img_dir = cfg["env"]["camera_outdir"]
        print("self.img_dir = ", self.img_dir)
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        self.resolution = 512

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.cfg["env"]["numObservations"] = 60
        self.cfg["env"]["numActions"] = 8

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # motion sequence to imitate; hard coded for now...
        self.motion_to_imitate = torch.tensor(
            np.loadtxt(self.cfg["motionToImitate"]),
            device=self.device
        )
        self.target_pose_idx = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)# + 301

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        #
        # Retrieves buffer for Actor root states. The buffer has shape 
        # (num_actors, 13). State for each actor root contains position([0:3]),
        # rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # Retrieves Degree-of-Freedom state buffer. Buffer has shape 
        # (num_dofs, 2). Each DOF state contains position and velocity.
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # Retrieves buffer for force sensors. The buffer has shape 
        # (num_force_sensors, 6). Each force sensor state has forces (3) and 
        # torques (3) data.
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 4
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # self.target is where the agent is running towards 
        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_ant.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(ant_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        body_names = [self.gym.get_asset_rigid_body_name(ant_asset, i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)

        # create force sensors attached to the "feet"
        extremity_indices = [self.gym.find_asset_rigid_body_index(ant_asset, name) for name in extremity_names]
        sensor_pose = gymapi.Transform()
        for body_idx in extremity_indices:
            self.gym.create_asset_force_sensor(ant_asset, body_idx, sensor_pose)

        self.ant_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        self.cams = []
        self.cam_tensors = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "ant", i, 1, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)

            if i < self.num_envs_to_visualize:

                print(f"Setting up cameras for visualizing env {i}...")

                # visualization logic copied from interop_torch.py

                # add camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.resolution
                cam_props.height = self.resolution
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                # global position (x, y, z), orientation
                self.gym.set_camera_location(cam_handle, env_ptr, gymapi.Vec3(15, 0, 3), gymapi.Vec3(0, 0, 0)) 
                self.cams.append(cam_handle)

                # obtain camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                print("  ...Got camera tensor with shape", cam_tensor.shape)

                # wrap camera tensor in a pytorch tensor
                torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(torch_cam_tensor)
                print("  ...Torch camera tensor device:", torch_cam_tensor.device)
                print("  ...Torch camera tensor shape:", torch_cam_tensor.shape)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles[0], extremity_names[i])


    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
            
        if self.viewer == None:

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            # fetch results
            # if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

            # render sensors and refresh camera tensors
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            # write out state and sensors periodically during the first little while
            if self.control_steps < self.num_frames_to_visualize:

                if self.control_steps % 10 == 0:
                    print("========= Saving frame %d ==========" % self.control_steps)

                for i in range(min(self.num_envs, self.num_envs_to_visualize)):

                    # write tensor to image
                    fname = os.path.join(self.img_dir, "cam-%04d-%04d.png" % (i, self.control_steps))
                    cam_img = self.cam_tensors[i].cpu().numpy()

                    cam_img = cv2.putText(
                        cam_img,
                        "target_pose_idx %d" % (self.target_pose_idx[i].item()),
                        (25, 480),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .4,
                        (255, 255, 255),
                    )

                    imageio.imwrite(fname, cam_img)


            self.gym.draw_viewer(self.viewer, self.sim, True)


    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.target_pose_idx[:] = compute_ant_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.motion_to_imitate,
            self.dof_limits_lower, 
            self.dof_limits_upper,
            self.target_pose_idx
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_ant_observations(
            self.obs_buf, self.root_states, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

    # Required for PBT training
    def compute_true_objective(self):

        velocity = self.root_states[:, 7:10]

        # We optimize for the maximum velocity along the x-axis (forward)
        self.extras['true_objective'] = velocity[:, 0].squeeze()

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, 2] = 0.0 # we don't care about the vector in the height direction
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        forces = self.actions * self.joint_gears * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        #  To manage actor behavior during simulation, you can apply DOF forces
        #  or PD controls using the following API:
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):

        # NATE ADDED THIS... NEED TO ADD RESET LOGIC...
        self.target_pose_idx += 1

        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.compute_true_objective()

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ant_reward(
    obs_buf, # [4096, 60]
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    termination_height,
    death_cost,
    max_episode_length,
    motion_to_imitate,
    dof_limits_lower,
    dof_limits_upper,
    target_pose_idx,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    # print("Where does backprop happen? Does it happen ONLY at the end of every episode? If so... need to handle longer episode lengths differently...")

    # IDEA: maybe we have a target joint angle for each frame/timestep? but we need to figure out 
    # how to make sure that the timesteps match up. ie for the given observation, we maybe need to
    # store the idx for the given frame of the motion sequence that we're supposed to be imitating.
    # we also need some way of resetting the agent if the target pose is too different from the
    # observation pose. procedure to accomplish this... ALL INSIDE REWARD FXN, DONT CHANGE OBSERVATION...
    # 
    #   1)  loading in the target motion sequence; say it's of length N
    #   2)  if reset flag for current agent is tripped, then start him at the
    #       first pose of the motion sequence (i.e. set target_pose_idx = 0 for
    #       that actor)
    #   3)  for each actor, look at target_motion_seq[target_pose_idx] to get the 8-dof
    #       vector that encodes how close he should be. if it's close enough, don't throw
    #       the reset flag, so we allow him to continue trying to learn the motion; if it's
    #       too far, then we throw the reset flag (see "Reset agents" logic), AND we add
    #       a death cost. 
    #   4)  note: the reward function shouldn't just be 1.5 - L2 distance, but instead should 
    #       be the exponential of the negative L2 distance, as laid out in PhysDiff
    #   5)  I think that we'll also need some global metric of how far along we are in the
    #       motion sequence. Maybe we should start with the MLP described in PhysDiff (to generate
    #       the gaussians from which we sample the actions, conditioned on observed states)
    #       but the problem with JUST that is that you only have access to the current state.
    #       so maybe for our purposes we might want some kind of way of accessing some representation
    #       of all previous states... the downside here is that it's not MARKOV anymore. I should
    #       bring this up in the txt2vid meeting, after confirming that generated motion isn't
    #       as strong. and don't move from ant to human till ive confirmed success in ant case.
    #       BUT OF COURSE: I need to start out w just the MDP...


    JUST_MEASURE_POSE_SIMILARITY = True

    if JUST_MEASURE_POSE_SIMILARITY:


        print(target_pose_idx[:3])
        # NATE ADDED ONLY THIS ONE...
        # computing joint angle similarity reward r_p^h from PhysDiff
        pose_to_imitate = motion_to_imitate[target_pose_idx] # [4096, 8]
        pose_scaled = unscale(pose_to_imitate, dof_limits_lower, dof_limits_upper) # [4096, 8]
        pose_diff = torch.sum(torch.abs(obs_buf[:, 12:20] - pose_scaled), dim=-1) # [4096]
        # print(torch.min(pose_diff), torch.mean(pose_diff), torch.max(pose_diff))
        # print(torch.mean(pose_diff))
        pose_reward = torch.exp(-pose_diff) # [4096]
        total_reward = pose_reward
        # print(target_pose_idx[:3])
        print(torch.mean(pose_diff).item(), torch.quantile(pose_diff, 0.99).item())

        # adjust reward for fallen agents
        total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)



        # reset agents
        reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
        # reset agents whose observed pose is too far away from the true pose
        QUANTILE_THRESHOLD = 0.99
        reset = torch.where(
            pose_diff > max(1.8, torch.quantile(pose_diff, QUANTILE_THRESHOLD)), # throw away agents with distances too large
            torch.ones_like(reset_buf), 
            reset)
        # adjust reward for agents whose observed pose is too far away from the true pose
        total_reward = torch.where(
            pose_diff > max(1.8, torch.quantile(pose_diff, QUANTILE_THRESHOLD)), 
            torch.ones_like(total_reward) * death_cost, 
            total_reward)

        # reset target pose in those cases
        target_pose_idx = torch.where(
            pose_diff > max(1.8, torch.quantile(pose_diff, QUANTILE_THRESHOLD)),
            torch.zeros_like(target_pose_idx),
            target_pose_idx
        )
        # when target_pose_idx is maxed out, the agent has survived for the whole motion sequence, 
        # so it deserves a reward
        total_reward = torch.where(
            target_pose_idx >= motion_to_imitate.shape[0] - 3, 
            -3.0 * torch.ones_like(total_reward) * death_cost, 
            total_reward)
        # then we need to reset those idxs to zero
        target_pose_idx = torch.where(
            target_pose_idx >= motion_to_imitate.shape[0] - 3,
            torch.zeros_like(target_pose_idx),
            target_pose_idx
        )

    else:

        print(target_pose_idx[:3])

        # energy penalty for movement
        actions_cost = torch.sum(actions ** 2, dim=-1)
        electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
        dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

        # reward for duration of staying alive
        alive_reward = torch.ones_like(potentials) * 0.5

        # NATE ADDED ONLY THIS ONE...
        # computing joint angle similarity reward r_p^h from PhysDiff
        pose_to_imitate = motion_to_imitate[target_pose_idx] # [4096, 8]
        pose_scaled = unscale(pose_to_imitate, dof_limits_lower, dof_limits_upper) # [4096, 8]
        pose_diff = torch.sum(torch.abs(obs_buf[:, 12:20] - pose_scaled), dim=-1) # [4096]
        # print(torch.min(pose_diff), torch.mean(pose_diff), torch.max(pose_diff))
        print(torch.mean(pose_diff))
        pose_reward = 20*torch.exp(-pose_diff) # [4096]
        total_reward = pose_reward

        # total_reward = pose_reward

        total_reward = alive_reward - \
            actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - \
            dof_at_limit_cost * joints_at_limit_cost_scale + \
            pose_reward

        # adjust reward for fallen agents
        total_reward = torch.where(
            obs_buf[:, 0] < termination_height, 
            torch.ones_like(total_reward) * death_cost, 
            total_reward
        )
        # print(torch.mean(total_reward)) # increases steadily!!

        # RESET AGENTS
        #
        # CASE 1: reset agents where root is too low (e.g. ant has flipped over)
        reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
        # CASE 2: reset agents whose episode has gone on longer than the max_episode_length
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
        # CASE 3: reset agents whose observed pose is too far away from the true pose
        # DIST_THRESHOLD = 0.95
        # THE QUANTILE RESULTED IN BAD PERFORMANCE; EPISODES DIDNT LAST LONG ENOUGH; NEED A QUANTIL SCHEDULE!!
        # reset = torch.where(
        #     pose_diff > torch.quantile(pose_diff, DIST_THRESHOLD), # only keep agents who aren't in the bottom DIST_THRESHOLD
        #     torch.ones_like(reset_buf), 
        #     reset)
        # # adjust reward for agents whose observed pose is too far away from the true pose
        # total_reward = torch.where(
        #     pose_diff > torch.quantile(pose_diff, DIST_THRESHOLD), 
        #     torch.ones_like(total_reward) * death_cost, 
        #     total_reward)        
        DIST_THRESHOLD = 2.0
        QUANTILE_THRESHOLD = 0.95
        reset = torch.where(
            pose_diff > max(DIST_THRESHOLD, torch.quantile(pose_diff, QUANTILE_THRESHOLD)), # throw away agents with distances too large
            torch.ones_like(reset_buf), 
            reset)
        # adjust reward for agents whose observed pose is too far away from the true pose
        total_reward = torch.where(
            pose_diff > max(DIST_THRESHOLD, torch.quantile(pose_diff, QUANTILE_THRESHOLD)), 
            torch.ones_like(total_reward) * death_cost, 
            total_reward)
        # print("PICK UP HERE... ALSO NEED TO ADJUST THE REWARD, NEGATIVELY FOR THOSE AGENTS!! see commented out section... without this bit, learning won't improve nearly as fast. SHOOULD (MUST!!) CHECK that mean difference indeed decreases!!!")

        # RESET TARGET POSE
        # 
        #
        # c.f. CASE 3; in that case, we need to set the corresponding target_pose_idx to 0
        # target_pose_idx = torch.where(
        #     pose_diff > torch.quantile(pose_diff, DIST_THRESHOLD),
        #     torch.zeros_like(target_pose_idx),
        #     target_pose_idx
        # )        
        target_pose_idx = torch.where(
            pose_diff > max(DIST_THRESHOLD, torch.quantile(pose_diff, QUANTILE_THRESHOLD)),
            torch.zeros_like(target_pose_idx),
            target_pose_idx
        )
        # when target_pose_idx is maxed out, the agent has survived for the whole motion sequence, 
        # so it deserves a reward; and then we need to reset those idxs to zero
        total_reward = torch.where(
            target_pose_idx >= motion_to_imitate.shape[0], 
            -1.0 * torch.ones_like(total_reward) * death_cost, 
            total_reward)
        target_pose_idx = torch.where(
            target_pose_idx >= motion_to_imitate.shape[0],
            torch.zeros_like(target_pose_idx),
            target_pose_idx
        )

        # for each agent:
        #   if at this current observation step, the agent's pose_diff is too far away:
        #       then set the corresponding target_pose_idx = 0, and reset_buf = 1




    """
    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
        
    # POSE REWARD, ADDED BY NATE
    pose = motion_to_imitate[301]
     # [lower, upper] -> [-1, 1]
    pose_scaled = unscale(pose, dof_limits_lower, dof_limits_upper)
    # reward for how close all 8 legs are to the target pose (just the ankle lifted...)
    pose_diff = torch.mean(torch.abs(obs_buf[:, 12:20] - pose_scaled), dim=-1)
    # I chose 1.5 after manually observing it's bigger than the max and mean
    pose_reward = 1.5 - pose_diff
    # print(torch.mean(pose_reward)) # watch it slowly increase ;)

    total_reward = alive_reward + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - \
        dof_at_limit_cost * joints_at_limit_cost_scale + \
        pose_reward
    # note--this simpler reward worked well, just less smooth
    # total_reward = pose_reward 

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    """

    return total_reward, reset, target_pose_idx


    


@torch.jit.script
def compute_ant_observations(obs_buf, root_states, targets, potentials,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale,
                             sensor_force_torques, actions, dt, contact_force_scale,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # to_target = vector pointing from root state to the goal destination, [1000,0,0]
    to_target = targets - torso_position 
    to_target[:, 2] = 0.0

    # print("FIGURE OUT WHAT POTENTIALS IS... AND IF IT'S NEEDED IN ORDER TO COMPUTE DISTANCE IN JOINT ANGLE SPACE")
    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)
    
    # unscale is a coordinate-wise function [lower, upper] -> [-1, 1]
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    obs = torch.cat((
        torso_position[:, up_axis_idx].view(-1, 1),                 # obs[0]        1
        vel_loc,                                                    # obs[1:4]      3
        angvel_loc,                                                 # obs[4:7]      3
        yaw.unsqueeze(-1),                                          # obs[7]        1
        roll.unsqueeze(-1),                                         # obs[8]        1
        angle_to_target.unsqueeze(-1),                              # obs[9]        1
        up_proj.unsqueeze(-1),                                      # obs[10]       1
        heading_proj.unsqueeze(-1),                                 # obs[11]       1
        dof_pos_scaled,                                             # obs[12:20]    8 = num_dofs
        dof_vel * dof_vel_scale,                                    # obs[20:28]    8 = num_dofs
        sensor_force_torques.view(-1, 24) * contact_force_scale,    # obs[28:52]    24
        actions),                                                   # obs[52:60]    8 = num_dofs
        dim=-1
    )

    return obs, potentials, prev_potentials_new, up_vec, heading_vec