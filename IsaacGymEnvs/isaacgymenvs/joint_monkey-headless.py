"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import imageio
import os
import cv2
import copy

import sys
sys.path.append("../../IsaacGymEnvs")
from isaacgymenvs.utils.torch_jit_utils import get_axis_params

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# simple asset descriptor for selecting from a list

def numpyify_state_positions(nested_tuples):

    # Initialize an empty list to store the concatenated elements
    concatenated_elements = []

    # Loop through each nested tuple in the list
    for outer_outer_tuple in nested_tuples:
        for outer_tuple in outer_outer_tuple:
            # Initialize an empty list to store the elements of each nested tuple
            row_elements = []
            # Loop through each sub-tuple in the nested tuple
            for inner_tuple in outer_tuple:
                # Extend the list with the elements of the sub-tuple
                row_elements.extend(inner_tuple)
            # Append this list to the main list
            concatenated_elements.append(np.array(row_elements))

    # Convert the list to a NumPy array with shape (22, 13)
    arr = np.array(concatenated_elements)
    arr = np.concatenate(arr, axis=-1).reshape(22, 13)

    return arr

class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

asset_descriptors = [
    AssetDesc("mjcf/nv_humanoid.xml", False),                                   # 0
    AssetDesc("mjcf/nv_ant.xml", False),                                        # 1
    AssetDesc("urdf/cartpole.urdf", False),                                     # 2
    AssetDesc("urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf", False),   # 3
    AssetDesc("urdf/franka_description/robots/franka_panda.urdf", True),        # 4
    AssetDesc("urdf/kinova_description/urdf/kinova.urdf", False),               # 5
    AssetDesc("urdf/anymal_b_simple_description/urdf/anymal.urdf", True),       # 6
    AssetDesc("mjcf/uhc.xml", False),                                           # 7
    AssetDesc("mjcf/uhc-edited.xml", False),                                    # 8
    AssetDesc("mjcf/amp_humanoid_sword_shield.xml", False),                     # 9
    AssetDesc("mjcf/uhc-v2.xml", False),                                        # 10
    AssetDesc("mjcf/uhc-v3.xml", False)                                         # 11
]


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
        {"name": "--headless", "action": "store_true", "help": ""},
        {"name": "--num_envs", "type": int, "default" : 36, "help": ""},
        {"name": "--resolution", "type": int, "default" : 128, "help": ""},
        {"name": "--save_motion_array", "type": str, "default" : '', "help": "Path to store motion as array"},        
        {"name": "--dof_to_save", "type": int, "default" : -1, "help": "If save_motion_array, then this you also need to choose which dof to save"},
        ]
    )

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
    quit()

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")


print("args.compute_device_id, args.graphics_device_id = ", args.compute_device_id, args.graphics_device_id)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer...
use_viewer = not args.headless
if use_viewer:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
else:
    viewer = None

# load asset
if args.asset_id <= 6:
    asset_root = "../../assets"
elif args.asset_id >= 7:
    asset_root = "../../IsaacGymEnvs/assets"
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# [CH] The code only models dof's

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)
num_dofs = gym.get_asset_dof_count(asset)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# [CH] display DOF's, features/properties describing each DOF, and state space 
print("\nDisplay DOF's by id, name and type...")
for i in range(num_dofs):
    print("  DOF{} {}: (type: {})".format(i,dof_names[i],dof_types[i]))

print("\nDisplay features describing each DOF...")
print(dof_props.shape)
for property in list(dof_props.dtype.names):
    print("  {}: (type: {})".format(property, dof_props[property].dtype ))
    # [CH] print(dof_props[property])

print("\nDisplay the state space...")
print(dof_states.shape)
for state in list(dof_states.dtype.names):
    print("  {}: (type: {})".format(state, dof_states[state].dtype ))
    # [CH] print(dof_states[state])

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array 
# [CH] why copy values? To ensure that hard-coded motions are achievable. 
stiffnesses = dof_props['stiffness']    # [CH] Not used in this program
dampings = dof_props['damping']         # [CH] Not used in this program
armatures = dof_props['armature']       # [CH] Not used in this program
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
# [CH] but elbows CAN have range [-4*pi, 4*pi] !?
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range 
        # [CH] setting default be extremum's !?
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0

    # set DOF position to default
    dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

'''
# [CH] copied arrays values are modified, the "dof_{}" named arrays stay the same !?
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])
'''

# set up the env grid
num_envs = args.num_envs
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
# cam_pos = gymapi.Vec3(17.2, 2.0, 16)
# cam_target = gymapi.Vec3(5, -2.5, 130)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []
cams = []
cam_tensors = []

# [CH] setup env + actor + camera
print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    # [CH] quaternion represents rotations
    pose = gymapi.Transform()
    if args.asset_id < 7 or args.asset_id == 9:
        pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107) 
    elif args.asset_id in [7, 8]:
        pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
        pose.r = gymapi.Quat(0.0, 0.707107, 0.0, 0.707107)

    elif args.asset_id in [11]:

        HEIGHT_INITIAL = 0.939
        # pose.p = gymapi.Vec3(0.0, HEIGHT_INITIAL, 0.0)
        # pose.r = gymapi.Quat(0.0, 0.707107, 0.0, 0.707107)

        pose.p = gymapi.Vec3(0.0, 0.0, HEIGHT_INITIAL)
        pose.r = gymapi.Quat(0.0, 0.707107, 0.70710, 0.0)

        # up_axis_idx = 2
        # pose.p = gymapi.Vec3(*get_axis_params(HEIGHT_INITIAL, up_axis_idx))
        # quat = "0.0 0.707107 0.70710 0.0"
        # quat = "0.0 0.0 0.0 1.0"
        # quat = [float(num) for num in str.split(quat)]
        # pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    # FOR VISUALIZING... COPIED FROM interop_torch

    # add camera
    cam_props = gymapi.CameraProperties()
    cam_props.width = args.resolution
    cam_props.height = args.resolution
    cam_props.enable_tensors = True
    cam_handle = gym.create_camera_sensor(env, cam_props)

    # cam_pos = gymapi.Vec3(2, 1, 0)
    # cam_target = gymapi.Vec3(0, 1, 0)    

    cam_pos = gymapi.Vec3(0, 2, 2)
    cam_target = gymapi.Vec3(0, 0, 1)

    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
    cams.append(cam_handle)

    # obtain camera tensor
    cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    print("Got camera tensor with shape", cam_tensor.shape)

    # wrap camera tensor in a pytorch tensor
    torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
    cam_tensors.append(torch_cam_tensor)
    print("  Torch camera tensor device:", torch_cam_tensor.device)
    print("  Torch camera tensor shape:", torch_cam_tensor.shape)


# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# initialize animation state
anim_state = ANIM_SEEK_LOWER
current_dof = 0 
print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

# create directory for saved images
img_dir = "../../outputs/joint_monkey_images"
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

MAX_FRAMES = math.inf
# MAX_FRAMES = 1300
frame_no = -1

if args.save_motion_array:
    dof_values = []     # <- element: dof_positions (63,), single dof's frame
    state_values = []   # <- element: state_positions (22,13), obtained by calling get_actor_rigid_body_states() 

# [CH] dof_positions[num_dof] is initiated the same as default[num_dof] values
# [CH] each dof_position[i] stores the current frame of each dof's !?
while viewer is None and frame_no < MAX_FRAMES:

    frame_no = gym.get_frame_count(sim) # 0, 1, 2, 3, ...

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    speed = speeds[current_dof]

    # animate the dofs
    # [CH] a sequence of motions from LOWER -> UPPER -> DEFAULT -> FINISHED
    if anim_state == ANIM_SEEK_LOWER:
        # starting at default position, first decrease position value until it
        # reaches its lower limit...
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= lower_limits[current_dof]:
            dof_positions[current_dof] = lower_limits[current_dof]
            anim_state = ANIM_SEEK_UPPER
    elif anim_state == ANIM_SEEK_UPPER:
        # then increase it unitl it reaches its upper limit...
        dof_positions[current_dof] += speed * dt
        if dof_positions[current_dof] >= upper_limits[current_dof]:
            dof_positions[current_dof] = upper_limits[current_dof]
            anim_state = ANIM_SEEK_DEFAULT
    if anim_state == ANIM_SEEK_DEFAULT:
        # then decrease it until it reaches its default value...
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= defaults[current_dof]:
            dof_positions[current_dof] = defaults[current_dof]
            anim_state = ANIM_FINISHED
    elif anim_state == ANIM_FINISHED:
        dof_positions[current_dof] = defaults[current_dof]

        if current_dof + 1 == num_dofs:
            print("Animated final DOF. Done!!")
            break
            # gym.destroy_viewer(viewer)
            # gym.destroy_sim(sim)

        current_dof = (current_dof + 1) % num_dofs
        anim_state = ANIM_SEEK_LOWER
        print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

        if args.save_motion_array and current_dof > args.dof_to_save:
            break


    if args.save_motion_array and current_dof != args.dof_to_save:
        continue
    else:
        # SAVE DOF VALUES FOR LATER...
        # [CH] 2D list, size (1,num_dof), this is just one frame, only care about the 55th dof, R_Shoulder_y
        dof_values.append(copy.deepcopy(dof_positions))
        # print(dof_positions.shape) # (63,)

    # [CH] not relevant, no viewer 
    if args.show_axis:
        gym.clear_lines(viewer)

    # [CH] retrieve actor rigid body info -> produces the same result. test case, 1/30/55
    # gym.refresh_rigid_body_state_tensor(sim)

    # clone actor state in all of the environments
    for i in range(num_envs):

        # [CH] all 0's at the beginning
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

        if args.save_motion_array and current_dof != args.dof_to_save:
            pass
        else:
            # SAVE STATE VALUES FOR LATER...
            # [CH] get_actor_rigid_body_states() -> (num_bodies,13), rigid body parts
            # [CH] rigid_body_states is read-only
            state_positions = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_ALL)
            # print(state_positions.shape) # [CH] (22,) 
            state_positions = numpyify_state_positions(state_positions) # 22, 13
            # print(state_positions.shape) # [CH] (22,13)
            state_values.append(copy.deepcopy(state_positions))
            
        # [CH] not relevant, no viewer
        if args.show_axis:
            # get the DOF frame (origin and axis)
            dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], current_dof)
            frame = gym.get_dof_frame(envs[i], dof_handle)

            # draw a line from DOF origin along the DOF axis
            p1 = frame.origin
            p2 = frame.origin + frame.axis * 0.7
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

    # rigid_body_states = gym.acquire_rigid_body_state_tensor(sim)
    # rigid_body_states = gym.get_actor_rigid_body_states(sim)
    # print(gymtorch.wrap_tensor(rigid_body_states).shape, gymtorch.wrap_tensor(rigid_body_states).shape, "\n\n\n\n") # 22, 13
    # quit()

    # update the viewer
    gym.step_graphics(sim)

    # render sensors and refresh camera tensors
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    # write out state and sensors periodically during the first little while
    if frame_no < math.inf:

        if frame_no % 100 == 0:
            print("========= Frame %d ==========" % frame_no)

        for i in range(num_envs):
            # write tensor to image
            fname = os.path.join(img_dir, "cam-%04d-%04d.png" % (i, frame_no))
            cam_img = cam_tensors[i].cpu().numpy()
            cam_img = cv2.putText(
                cam_img,
                "DOF %d ('%s')" % (current_dof, dof_names[current_dof]),
                (25, 480),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (255, 255, 255),
            )
            imageio.imwrite(fname, cam_img)

    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

if args.save_motion_array:

    # CASE 1: the motion "sequence" is really just one pose...
    # pose_to_clone = copy.deepcopy(dof_values[-1])
    # for _ in range(400):
    #     dof_values.append(pose_to_clone)
    # dof_values = np.concatenate(dof_values)
    # dof_values = np.reshape(dof_values, (-1, 8))
    # np.savetxt(args.save_motion_array, dof_values)

    # CASE 2: a simple motion sequence that's an actual sequence
    dof_values = np.concatenate(dof_values)
    dof_values = np.reshape(dof_values, (-1, num_dofs)) # 160, 63
    print(dof_values.shape)

    state_values = np.concatenate(state_values)
    state_values = np.reshape(state_values, (-1, 22*13)) # 160, 286
    print(state_values.shape)

    motion_arr = np.concatenate((dof_values, state_values), axis=1) # [CH] 160, 349
    print(motion_arr.shape)
    np.savetxt(args.save_motion_array, motion_arr)



    # CASE 3: literally just save what's been viewed...
    # dof_values = np.concatenate(dof_values)
    # dof_values = np.reshape(dof_values, (-1, 8))
    # np.savetxt(args.save_motion_array, dof_values)

'''[CH]
Call refresh_{}(sim) in headless mode
# self.gym.refresh_dof_state_tensor(self.sim)
# self.gym.refresh_actor_root_state_tensor(self.sim)
# self.gym.refresh_rigid_body_state_tensor(self.sim)  #THIS!
# self.gym.refresh_force_sensor_tensor(self.sim)
# self.gym.refresh_dof_force_tensor(self.sim)
'''

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
