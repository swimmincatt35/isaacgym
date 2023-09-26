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
from os.path import join
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, \
    quat_conjugate, quat_from_angle_axis, to_torch, get_axis_params, \
    torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, \
    normalize_angle, extract_positions_from_rigid_body_states

from isaacgymenvs.tasks.base.vec_task import VecTask
from tqdm import tqdm
import sys
import imageio
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


class HumanoidImitate(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # visualization logic, added by Nate
        self.num_envs_to_visualize = 1
        self.num_frames_to_visualize = math.inf
        self.img_dir = cfg["env"]["camera_outdir"]
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        self.resolution = cfg["env"]["camera_resolution"]
        self.render_every = 10000
        self.render_for = 1000
        
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.termination_radius = self.cfg["env"]["terminationRadius"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.debug_viz_rigid_bodies = self.cfg["env"]["enableDebugVisRigidBodies"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # we change these values manually, dervied using the values in the xml file
        self.cfg["env"]["numObservations"] = 205
        self.cfg["env"]["numActions"] = 63 # 3 * 21

        super().__init__(
            config=self.cfg, 
            rl_device=rl_device, 
            sim_device=sim_device, 
            graphics_device_id=graphics_device_id, 
            headless=headless, 
            virtual_screen_capture=virtual_screen_capture, 
            force_render=force_render
        )

        # the pose we want to imitate
        self.motion_to_imitate = torch.tensor(
            # np.loadtxt("../../assets/motions/humanoid/move_left_elbow.txt"),
            np.loadtxt("../../assets/motions/humanoid/move_left_arm.txt"),
            device=self.device
        )

        # get gym GPU state tensors
        #
        # Retrieves buffer for Actor root states. The buffer has shape 
        # (num_actors, 13). State for each actor root contains position([0:3]),
        # rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)   # 4096, 13
        # Retrieves Degree-of-Freedom state buffer. Buffer has shape 
        # (num_envs * num_dofs, 2). Each DOF state contains position and velocity.
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)          # 4096 * 63, 2
        # Retrieves buffer for force sensors. The buffer has shape 
        # (num_envs * num_force_sensors, 6). Each force sensor state has forces (3) and 
        # torques (3) data.
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)          # 4096 * 2, 6
        # Retrieves buffer for Rigid body states. The buffer has shape (num_rigid_bodies, 13). 
        # State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), 
        # and angular velocity([10:13]).
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1) # 4096, 22*13

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0

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

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        print(f"self.device_id, self.graphics_device_id =", self.device_id, self.graphics_device_id)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", False)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "R_Ankle")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "L_Ankle")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        HEIGHT_INITIAL = 0.939
        start_pose.p = gymapi.Vec3(*get_axis_params(HEIGHT_INITIAL, self.up_axis_idx))  # 0.0 0.0 0.939
        quat = "0.0 0.707107 0.70710 0.0"
        # quat = "0.0 0.0 0.0 1.0"
        quat = [float(num) for num in str.split(quat)]
        start_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0, 0, -1))

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        self.cams = []
        self.cam_tensors = []

        print("creating envs...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)


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
                # for looking at origin up close at foot level
                # camera_position = gymapi.Vec3(0, 0.3, 0.03)
                # camera_target = gymapi.Vec3(0, 0, 0.03)
                # for looking at origin from afar
                camera_position = gymapi.Vec3(0, 2, 2)
                camera_target = gymapi.Vec3(0, 0, 1)
                self.gym.set_camera_location(cam_handle, env_ptr, camera_position, camera_target)
                self.cams.append(cam_handle)

                # obtain camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                print("   ...Got camera tensor with shape", cam_tensor.shape)

                # wrap camera tensor in a pytorch tensor
                torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(torch_cam_tensor)
                print("  ...Torch camera tensor device:", torch_cam_tensor.device)
                print("  ...Torch camera tensor shape:", torch_cam_tensor.shape)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
            
        if self.cfg["env"]["render_frames"] == True:

            if self.control_steps % self.render_every not in range(self.render_for):
                return None

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
                    fname = os.path.join(self.img_dir, "cam-%04d-%08d.png" % (i, self.control_steps))
                    cam_img = self.cam_tensors[i].cpu().numpy()
                    imageio.imwrite(fname, cam_img)

            self.gym.draw_viewer(self.viewer, self.sim, True)

    def compute_reward(self, actions):

        """
        self.rew_buf[:], self.reset_buf = compute_humanoid_reward(
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
            self.max_motor_effort,
            self.motor_efforts,
            self.termination_height,
            self.death_cost,
            self.max_episode_length
        )
        """

        self.rew_buf[:], self.reset_buf = compute_humanoid_reward(
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
            self.max_motor_effort,
            self.motor_efforts,
            self.termination_height,
            self.termination_radius,
            self.death_cost,
            self.max_episode_length,
            self.motion_to_imitate,
            self.dof_limits_lower, 
            self.dof_limits_upper
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        """
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_humanoid_observations(
            self.obs_buf, self.root_states, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1)
        """
        
        """
        _ = compute_humanoid_observations(
            self.obs_buf,                   # [4096, 108]
            self.root_states,               # [4096, 13]
            self.targets,                   # [4096, 3]
            self.potentials,                # [4096]
            self.inv_start_rot,             # [4096, 4]
            self.dof_pos,                   # [4096, 63]
            self.dof_vel,                   # [4096, 63]
            self.dof_force_tensor,          # [4096, 63]
            self.dof_limits_lower,          # [63]
            self.dof_limits_upper,          # [63]
            self.dof_vel_scale,             # 0.1
            self.vec_sensor_tensor,         # [4096, 12]
            self.actions,                   # [4096, 63]
            self.dt,                        # 0.0166
            self.contact_force_scale,       # 0.01
            self.angular_velocity_scale,    # 0.25
            self.basis_vec0,                # [4096, 3]
            self.basis_vec1                 # [4096, 3]
        )
        """

        self.obs_buf[:] = compute_humanoid_observations(
            self.dof_pos,                   # [4096, 63]
            self.dof_limits_lower,          # [63]
            self.dof_limits_upper,          # [63],
            self.dof_vel,                   # [4096, 63]
            self.dof_vel_scale,             # 0.1,
            self.root_states,               # [4096, 13]
            self.rigid_body_states          # [4096, 22*13]
        )


    def plot_3d_motion_frame(
        self, 
        kinematic_chain,
        tgt_ani_dir,
        rigid_body_coords,
        chain_type
    ):

        self.control_steps
        save_path = join(tgt_ani_dir, os.path.join(self.img_dir, f"chains-{chain_type}-%08d.png" % (self.control_steps-1)))

        if chain_type in ["gt", "sim"] and self.control_steps < 2:
            return None # for some reason, I can't obtain the first two simulator states
        if chain_type == "gt" and self.control_steps > 60:
            return None # static image... no need
    
        save_path, 
        kinematic_tree = kinematic_chain
        joints = rigid_body_coords
        figsize=(10, 10)
        radius=4

        def init():
            ax.set_xlim3d([-radius / 2, radius / 2]) # x-axis: to the right
            ax.set_ylim3d([-radius / 2, radius / 2]) # y-axis: towards you
            ax.set_zlim3d([0, radius]) # z-axis: up
            ax.grid(b=False)

        def plot_xyPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        def draw_axes(length=radius):
            # Draw x-axis
            ax.plot([0, length], [0, 0], [0, 0], color='r', label='X')
            ax.text(length/2, 0, 0, 'X', color='r', fontsize=32)
            # Draw y-axis (towards you)
            ax.plot([0, 0], [0, -length], [0, 0], color='g', label='Y')
            ax.text(0, -length/2, 0, 'Y', color='g', fontsize=32)
            # Draw z-axis (upwards)
            ax.plot([0, 0], [0, 0], [0, length], color='b', label='Z')
            ax.text(0, 0, length/2, 'Z', color='b', fontsize=32)



        # (seq_len, joints_num, 3) = (116, 22, 3)
       
        data = joints.copy().reshape(len(joints), -1, 3) # doesnt change anything for 000000
        fig = plt.figure(figsize=figsize)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        # print("data:", data)
        init()
        MINS = data.min(axis=0).min(axis=0) # array([-0.50, 0.0, -0.32], dtype=float32)... so y is height?
        MAXS = data.max(axis=0).max(axis=0)
        colors = ['darkred', 'darkblue', 'black', 'red', 'blue',  
                'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
                'darkred', 'darkred','darkred','darkred','darkred'] # only first 5 colors are used
        frame_number = data.shape[1]

        # changes all heights by the offset so floor is at 0

        if chain_type == "sim":
            height_offset = MINS[2]
            # data[:, :, 2] -= height_offset
            # (116, 2)... just the (x,z) coordinates [ie ground coordinates] of the first joint (root joint)
            # I think this is what gets plotted underneath the motion

        trajec = data[:, 0, [0, 1]] 
        
        # normalizing the x coordinate of the data by removing the x coordinates of
        # the root joint (ie the first joint) from all the x coordinates of the data
        # data[..., 0] -= data[:, 0:1, 0] # (116, 22) - (116, 1) = (116, 22)
        # doing the same with the z coordinate
        # data[..., 1] -= data[:, 0:1, 1] # (116, 22) - (116, 1) = (116, 22)
        # the result is a data.shape = (116, 22, 3) just like before, but with the x and z coordinates
        # (i.e. the projection onto the ground coordinates) starting at (0, 0) for EVERY timestep for the root joint.
        # this just means we're centering the motion around the root joint. Notice we can recover this lost
        # information using trajec, which we saved above. (Take a look at the saved indices and the subtracted
        # values; they're the same!!!)

        # if chain_type == "sim":
        #     data[..., 2] = -data[..., 2]


        # print("data:", data)

        def update(index):
            ax.lines = []
            ax.collections = []
            ax.view_init(elev=45, azim=-90)
            ax.dist = 7.5
            plot_xyPlane(
                MINS[0]-trajec[index, 0], 
                MAXS[0]-trajec[index, 0], 
                0, 
                MINS[1]-trajec[index, 1], 
                MAXS[1]-trajec[index, 1]
            )
            
            # if index > 1:
            #     ax.plot3D(trajec[:index, 0]-trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1]-trajec[index, 1], linewidth=1.0,
            #             color='blue')
            
            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                linewidth = 4.0
                ax.plot3D(data[chain, index, 0], data[chain, index, 1], data[chain, index, 2], linewidth=linewidth, color=color)

            # Draw the axes
            draw_axes()

            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/20, repeat=False)

        ani.save(save_path)
        plt.close()



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
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):

        if self.debug_viz_rigid_bodies and self.control_steps % self.render_every in range(self.render_for):

            # STEP 1: plot self.rigid_body_states[0], which has shape [22*13]; only use the first 3
            # rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.humanoid_handles[0])
            kinematic_chain = [
                [0, 5, 6, 7, 8],        # Pelvis, R_Hip, R_Knee, R_Ankle, R_Toe
                [0, 1, 2, 3, 4],        # Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe
                [0, 9, 10, 11, 12, 13], # Pelvis, Torso, Spine, Chest, Neck, Head
                [11, 18, 19, 20, 21],   # Chest, R_Thorax, R_Shoulder, R_Elbow, R_Wrist
                [11, 14, 15, 16, 17],   # Chest, L_Thorax, L_Shoulder, L_Elbow, L_Wrist
            ]
            rigid_body_coords = np.asarray(self.rigid_body_states[0].reshape(22,13)[:,:3].cpu()) # 22, 3
            self.plot_3d_motion_frame(kinematic_chain, self.img_dir, rigid_body_coords, "sim")

            # STEP 2: plot motion_to_imitate[40, :63] along with self.root_states, which has shape [13]
            rigid_body_coords = np.asarray(self.motion_to_imitate[41, 63:].reshape(22,13)[:,:3].cpu()) # 22, 3
            self.plot_3d_motion_frame(kinematic_chain, self.img_dir, rigid_body_coords, "gt")

        self.actions = actions.to(self.device).clone()
        # print(self.actions.shape, self.motor_efforts.unsqueeze(0).shape, self.power_scale) 
        # torch.Size([4096, 63]), torch.Size([1, 63]) 1.0
        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        """
        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

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
        """

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_humanoid_reward(
    obs_buf,
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
    max_motor_effort,
    motor_efforts,
    termination_height,
    termination_radius,
    death_cost,
    max_episode_length,
    motion_to_imitate, 
    dof_limits_lower, 
    dof_limits_upper
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, Tensor, float, float, float, float, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # POSE REWARD, ADDED BY NATE
    pose = motion_to_imitate[41, :63]
    pose_scaled = unscale(pose, dof_limits_lower, dof_limits_upper) # [lower, upper] -> [-1, 1]
    # print(pose_scaled)
    # print(pose_scaled[46])
    # print(obs_buf[0,46])
    pose_diff = torch.sum(torch.abs(obs_buf[:, 0:63] - pose_scaled[0:63]), dim=-1) # elbow bent
    # pose_diff = torch.sum(torch.abs(obs_buf[:, 0:63] - 0.0), dim=-1) # straight pose
    reward_pose = torch.exp(-0.3 * pose_diff)
    # print("Mean pose reward: ", torch.mean(reward_pose).item()) # watch it slowly increase ;)

    pose_position = extract_positions_from_rigid_body_states(motion_to_imitate[41, 63:].unsqueeze(0))
    rigid_body_position = obs_buf[:, 139:205]
    rigid_body_difference = torch.sum(torch.abs(rigid_body_position-pose_position), dim=-1) # elbow bent
    # rigid_body_difference = torch.sum(torch.abs(rigid_body_position-0.0), dim=-1) # just standing up straight...
    reward_rigid_body_position = torch.exp(-0.2 * rigid_body_difference)


    actions_cost = torch.sum(actions ** 2, dim=-1)
    # energy cost reward
    motor_effort_ratio = motor_efforts / max_motor_effort
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 0:63]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 0:63]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 63:126]) * motor_effort_ratio.unsqueeze(0), dim=-1)


    reward_velocity = torch.exp(-0.05 * torch.sum(torch.abs(obs_buf[:, 63:126]), dim=-1))


    # reward for being upright
    root_above_termination_height = obs_buf[:, 128] - termination_height
    reward_up = torch.where(
        root_above_termination_height > 0.0, 
        up_weight + torch.zeros_like(pose_diff),
        torch.zeros_like(pose_diff)
    )
    # reward for being close enough to origin
    dist_from_origin = torch.sqrt(obs_buf[:, 126]**2 + obs_buf[:, 127]**2)
    reward_centered = torch.exp(-dist_from_origin)


    rewards_to_include = {}
    rewards_to_include["1-reward_rigid_body_position"]  = reward_rigid_body_position
    rewards_to_include["2-reward_pose"]                 = reward_pose
    rewards_to_include["3-reward_velocity"]             = reward_velocity
    rewards_to_include["4-reward_up"]                   = reward_up
    rewards_to_include["5-reward_centered"]             = 0.1*reward_centered
    rewards_to_include["6-cost_actions"]                = - actions_cost_scale * actions_cost
    # rewards_to_include["7-cost_dof_at_limits"]          = - dof_at_limit_cost * joints_at_limit_cost_scale

    total_reward = torch.zeros_like(pose_diff)
    rewards_str = ""
    for reward_name in rewards_to_include:
        rewards_str += str(torch.mean(rewards_to_include[reward_name]).item()) + " "
        total_reward += rewards_to_include[reward_name]
    print(str(torch.mean(total_reward).item()) + "     " + rewards_str)
    
    reset = torch.where(obs_buf[:, 128] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(obs_buf[:, 126]**2 + obs_buf[:, 127]**2 > termination_radius**2, torch.ones_like(reset), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset), reset)

    return total_reward, reset


    # reward from the direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # reward for being upright
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    actions_cost = torch.sum(actions ** 2, dim=-1)

    # energy cost reward
    motor_effort_ratio = motor_efforts / max_motor_effort
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 12:33]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)

    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 33:54]) * motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of being alive
    alive_reward = torch.ones_like(potentials) * 2.0
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


@torch.jit.script
def compute_humanoid_observations(
        dof_pos,
        dof_limits_lower,
        dof_limits_upper,
        dof_vel,
        dof_vel_scale,
        root_states,
        rigid_body_states
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor) -> Tensor
    # print("dof_pos.shape:", dof_pos.shape, "\n\n\n\n\n")

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # print(rigid_body_states.shape, rigid_body_states[0][:26])
    rigid_body_positions = extract_positions_from_rigid_body_states(rigid_body_states) # [4096, 66]
    # print(positions_xyz.shape, positions_xyz[0][:26])

    # obs = dof_pos_scaled

    dof_vel_scaled = dof_vel * dof_vel_scale
    # print(dof_vel_scaled.shape)

    # print() # fixes a bug...
    obs = torch.cat((
        dof_pos_scaled,                     # obs[:, 0:63]
        dof_vel_scaled,                     # obs[:, 63:126]
        root_states,                        # obs[:, 126:139]; position([0:3]), rotation([3:7]), linear velocity([7:10]),angular velocity([10:13]).
        rigid_body_positions                # obs[:, 139:205]; x_0,y_0,z_0,...,x_21,y_21,z_21
    ), dim=-1)

    # print("GOT HERE 1\n\n\n")
    # print(obs.shape)


    return obs

"""
def compute_humanoid_observations(
        obs_buf, 
        root_states, 
        targets, 
        potentials, 
        inv_start_rot, 
        dof_pos, 
        dof_vel,
        dof_force, 
        dof_limits_lower, 
        dof_limits_upper, 
        dof_vel_scale,
        sensor_force_torques, 
        actions, 
        dt, 
        contact_force_scale, 
        angular_velocity_scale,
        basis_vec0, 
        basis_vec1):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (21), num_dofs (21), 6, num_acts (21)
    obs = torch.cat((torso_position[:, 2].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                     yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
                     sensor_force_torques.view(-1, 12) * contact_force_scale, actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec
"""
