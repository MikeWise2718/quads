import time
import numpy as np
import carb

from pxr import UsdPhysics, Usd, UsdGeom, Gf

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World

from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats

from .senut import add_sphere_light_to_stage
from .scenario_base import ScenarioBase

from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage
from omni.isaac.quadruped.robots import Unitree


# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

class QuadrupedScenario(ScenarioBase):
    _running_scenario = False
    _show_collision_bounds = True

    def __init__(self, uibuilder=None):
        super().__init__()
        self._scenario_name = "quadruped"
        self._scenario_description = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None
        self._nrobots = 1
        self.uibuilder = uibuilder

    def load_scenario(self, robot_name, ground_opt):
        super().load_scenario(robot_name, ground_opt)

        self._world_settings = {}
        self._world_settings["stage_units_in_meters"] = 1.0
        # self._world_settings["physics_dt"] = 1.0 / 400.0
        # self._world_settings["rendering_dt"] = 5.0 / 400.0
        self._world_settings["physics_dt"] = 1.0 / 60.0
        self._world_settings["rendering_dt"] = 5.0 / 400.0

        world = World.instance()

        self._base_command = [0.0, 0.0, 0.0, 0]
        self._event_flag = False

        # bindings for keyboard to command
        self._command_mapping = {
            # forward command
            "Forward": [15, 0.0, 0.0],
            # back command
            "Back": [-15, 0.0, 0.0],
            # left command
            "Left": [0.0, -4.0, 0.0],
            # right command
            "Right": [0.0, 4.0, 0.0],
            # yaw command (positive)
            "Yaw-plus": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "Yaw-neg": [0.0, 0.0, -1.0],
        }


        # self._ro bcfg = self.create_robot_config(robot_name, ground_opt)

        self.add_light("sphere_light")
        self.add_ground(ground_opt)

        self.create_robot_config(robot_name, "/World/roborg", ground_opt)
        # self._robcfg = self.get_robot_config()
        # self.load_robot_into_scene()

        match robot_name:
            case "a1":
                self._a1 = world.scene.add(
                    Unitree(
                        prim_path="/World/A1",
                        name="A1",
                        model="A1",
                        position=np.array([0, 0, -0.700]),
                        physics_dt=self._world_settings["physics_dt"],
                    )
                )
            case "spot":
                self._a1 = world.scene.add(
                    Unitree(
                        prim_path="/World/Spot",
                        name="Spot",
                        model="Spot",
                        position=np.array([0, 1, -0.400]),
                        physics_dt=self._world_settings["physics_dt"],
                    )
                )

        self.phystep = 0
        self.ikerrs = 0

        self._robot_name = robot_name
        self._ground_opt = ground_opt
        self._stage = get_current_stage()
        self._world = world

        self.add_light("sphere_light")
        self.add_ground(ground_opt)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        sz = 0.04
        self._target = XFormPrim("/World/target", scale=[sz, sz, sz])

    def post_load_scenario(self):
        print("Quadruped post_load_scenario")

        # self.register_robot_articulations()
        # self.teleport_robots_to_zeropos()

        # RMPflow config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"

        # rcfg = self.get_robot_config()
        # rcfg._kinematics_solver = LulaKinematicsSolver(
        #     robot_description_path = rcfg.rdf_path,
        #     urdf_path = rcfg.urdf_path
        # )

        # eename = rcfg.eeframe_name
        # rcfg._articulation_kinematics_solver = ArticulationKinematicsSolver(rcfg._articulation, rcfg._kinematics_solver, eename)
        # ee_position, ee_rot_mat = rcfg._articulation_kinematics_solver.compute_end_effector_pose()
        # self._ee_pos = ee_position
        # self._ee_rot = ee_rot_mat

        # print("Valid frame names at which to compute kinematics:", rcfg._kinematics_solver.get_all_frame_names())

    async def reset_scenario(self):

        await self._world.play_async()
        self._a1.set_state(self._a1._default_a1_state)
        self._a1.post_reset()
        print("Quadruped reset_scenario done")
        # self.teleport_robots_to_zeropos()

        # rcfg = self.get_robot_config()

        # ee_position, ee_rot_mat = rcfg._articulation_kinematics_solver.compute_end_effector_pose()
        # self._ee_pos = ee_position
        # self._ee_rot = ee_rot_mat

        # self._target.set_world_pose(self._ee_pos, rot_matrices_to_quats(self._ee_rot))

    phystep = 0
    ikerrs = 0
    ik_solving_active = True
    msggap = 1
    last_msg_time = 0

    def physics_step(self, step_size):
        # rcfg = self.get_robot_config()

        self.phystep += 1

        if self._event_flag:
            self._a1._qp_controller.switch_mode()
            self._event_flag = False

        now = time.time()
        if now - self.last_msg_time > self.msggap:
            # print("physics step - step size:", step_size," base_command:", self._base_command)
            print(f"phystep:{self.phystep}  step_size: {step_size:.5f}  base_cmd: {self._base_command}")
            self.last_msg_time = now
        self._a1.advance(step_size, self._base_command)

        # ee_position, ee_rot_mat = rcfg._articulation_kinematics_solver.compute_end_effector_pose()
        # self._ee_pos = ee_position
        # self._ee_rot = ee_rot_mat

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)

    def scenario_action(self, actionname, mouse_button=0 ):
        print("QuadrupedScenario action:", actionname, "   mouse_button:", mouse_button)
        if actionname.startswith("Cmd:"):
            _,key = actionname.split(":")
            if key in self._command_mapping:
                self._base_command[0:3] = self._command_mapping[key]
                self._event_flag = True
        elif actionname == "Nop":
            pass
        else:
            print(f"Unknown actionname: {actionname}")

    def get_scenario_actions(self):
        rv = ["Cmd:Forward", "Cmd:Back", "Cmd:Left", "Cmd:Right", "Cmd:Yaw-plus", "Cmd:Yaw-neg"]
        return rv
