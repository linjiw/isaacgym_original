import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os

from isaacgym import gymutil, gymapi
from isaacgym.terrain_utils import *
from math import sqrt


class TerrainSimulation:
    def __init__(self, args):
        self.gym = gymapi.acquire_gym()
        self.args = args
        self.sim = None
        self.viewer = None
        self.envs = []
        self.initial_state = None
        self.asset = None
        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets")
        self.setup_simulation()

    def setup_simulation(self):
        # Configure simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.substeps = 2
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = self.args.num_threads
        sim_params.physx.use_gpu = self.args.use_gpu
        
        # Create simulation
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise ValueError("*** Failed to create sim")
        
        # Load assets and create environments
        self.load_assets_and_create_envs()

        # Create viewer
        self.create_viewer()

    def load_assets_and_create_envs(self):
        # Load ball asset
        asset_file = "urdf/ball.urdf"
        self.asset = self.gym.load_asset(self.sim, self.asset_root, asset_file, gymapi.AssetOptions())
        
        # Create environments
        num_envs = 1
        num_per_row = 80
        env_spacing = 0.56
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        pose = gymapi.Transform()
        pose.p.z = 1.0
        
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            ahandle = self.gym.create_actor(env, self.asset, pose, "ball", i, 1)
            # Set random colors, etc.

        # Capture the initial state for resets
        self.initial_state = self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL)

    def create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError("*** Failed to create viewer")
        
        cam_pos = gymapi.Vec3(-5, -5, 15)
        cam_target = gymapi.Vec3(0, 0, 10)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Subscribe to keyboard events
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

    def run(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Handle keyboard events, simulate, fetch results, and update the viewer
            pass  # Implement the simulation loop based on your needs

    def reset_simulation(self):
        if self.initial_state is not None:
            self.gym.set_sim_rigid_body_states(self.sim, self.initial_state, gymapi.STATE_ALL)

    def destroy(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None

    def recreate_sim(self):
        self.destroy()
        self.setup_simulation()


# Parse arguments
args = gymutil.parse_arguments()

# Initialize the simulation
terrain_sim = TerrainSimulation(args)

# Run the simulation
terrain_sim.run()

# Optionally reset or recreate the simulation
terrain_sim.reset_simulation()
# or
terrain_sim.recreate_sim()
