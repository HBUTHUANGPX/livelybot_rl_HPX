# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch

# 串联pai
# 完全竖直时 baselink height = 0.36113 foot height=0.36113-0.3105 = 0.05063
"""
# 三个关节弯曲角度为 -0.25 0.65 -0.4 时 baselink height = 0.3453  foot height=0.05063

"""


class Pai_Cl_cycle:
    # default_device = "cpu"
    default_device = "cuda:0"
    def __init__(self, phase, side="l") -> None:
        self.phase: torch.Tensor = phase
        # print("2",self.phase[0])
        self.side = side
        self.T = 0.3  # 步态周期
        self.beta = 0.5  # 站姿相位的比例因子
        self.omega = 0.0  # 角速度
        self.lx = 1.0  # 腿长
        self.k_i = 1.0  # 常数
        self.h = 0.03  # 抬高高度
        self.ref_dof_pos = torch.zeros((self.phase.size(dim=0), 6), device=self.default_device)
        self.foot_end_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.default_device)
        self.phyi = torch.zeros_like(self.phase, device=self.default_device)
        self.pi = torch.zeros_like(self.phase, device=self.default_device)
        self.vx = torch.ones_like(self.phase, device=self.default_device)  # x 方向速度
        self.vy = torch.zeros_like(self.phase, device=self.default_device)  # y 方向速度
        self.first_step = torch.zeros_like(self.phase, device=self.default_device)
        if self.side == "l":
            self.l0 = [-5.1979e-05, 0.0233, -0.033]
            self.l1 = [0, 0.0568, 0]
            self.l2 = [0, 0, -0.06925]
            self.l3 = [0, 0, -0.07025]
            self.l4 = [0, 0, -0.14]
            self.l5 = [0.07525, 0, 0]
            self.start_phy = 0. * torch.pi
        elif self.side == "r":
            self.l0 = [-5.1979e-05, -0.0233, -0.033]
            self.l1 = [0.00025, -0.0568, 0]
            self.l2 = [-0.00025, 0, -0.06925]
            self.l3 = [0, -0.0027, -0.07025]
            self.l4 = [0, 0, -0.14]
            self.l5 = [0.07525, 0.0027, 0]
            self.start_phy = 1.0 * torch.pi
        # self.broadcaster = tf2_ros.TransformBroadcaster()
        # self.tfs = TransformStamped()

    def compute(self):
        self.init_pos()
        self.fresh_joint_angle_with_ref()
        self.compute_r_i(self.start_phy)
        self.fresh_ref_joint_angle_with_cycle()
        self.fresh_joint_angle_with_ref()

    def init_pos(self):
        self.ref_dof_pos[:, 0] = -0.25
        self.ref_dof_pos[:, 3] = 0.65
        self.ref_dof_pos[:, 4] = -0.4
        self.j0 = self.ref_dof_pos[:, 0]
        self.j3 = self.ref_dof_pos[:, 3]
        self.y = torch.full_like(self.j0, 0.0801)  # y 分量是常数

    def fresh_joint_angle_with_ref(self):
        # 计算 x, y, z, w 分量
        self.x = (
            0.1395 * torch.sin(self.j0)
            + 0.14 * torch.sin(self.j0 + self.j3)
            - 5.1979e-5
        )
        self.z = -0.1395 * torch.cos(self.j0) - 0.14 * torch.cos(
            self.j0 + self.j3
        )

    def compute_p_i(self):
        phi_i = (self.phase % self.T) / self.T * 2 * torch.pi
        return phi_i

    def compute_r_i(self, start_phase=torch.pi):
        # print("3",self.phase[0])
        self.phyi = (self.phase % self.T) / self.T * 2 * torch.pi + start_phase

        self.phyi[self.phyi > torch.pi] -= 2 * torch.pi

        self.pi = torch.abs(self.phyi) / torch.pi
        pi_cubed = self.pi**3
        pi_fourth = pi_cubed * self.pi
        pi_fifth = pi_fourth * self.pi
        pi_sixth = pi_fifth * self.pi
        poly_x_y = 6 * pi_fifth - 15 * pi_fourth + 10 * pi_cubed - 0.5-0.03
        poly_z = -64 * pi_sixth + 192 * pi_fifth - 192 * pi_fourth + 64 * pi_cubed
        # print("Tensor on GPU:", self.x[0],self.vx[0],self.T,self.beta,poly_x_y[0],self.phyi[0])
        self.r_i_x = self.x + self.vx * self.T * self.beta * poly_x_y

        # 计算 r_i_y
        # self.r_i_y = (
        #     self.y
        #     + (self.vy + self.k_i * self.omega * self.lx / 2)
        #     * self.T
        #     * self.beta
        #     * poly_x_y
        # )

        # 计算 r_i_z
        self.r_i_z = self.z + self.h * poly_z

        self.r_i_z[self.phyi < 0] = self.z[self.phyi < 0]

    def fresh_ref_joint_angle_with_cycle(self):
        l1 = self.l2[2] + self.l3[2]
        l2 = self.l4[2]
        d = (self.r_i_x**2 + self.r_i_z**2 - l1**2 - l2**2) / (2 * l1 * l2)
        sita2 = torch.atan2(torch.sqrt(1 - d**2), d)
        sita2[sita2 < -torch.pi] += 2 * torch.pi
        sita1 = torch.atan2(
            l2 * torch.sin(sita2), l1 + l2 * torch.cos(sita2)
        ) - torch.atan2(self.r_i_x, self.r_i_z)
        sita1[sita1 < -torch.pi] += 2 * torch.pi
        self.ref_dof_pos[:, 0] = -sita1  # + 0.25
        self.ref_dof_pos[:, 3] = sita2  # - 0.65
        self.ref_dof_pos[:, 4] = -sita2 + sita1  # + 0.4



class Pai_Cl_0905_Cfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * (num_single_obs))
        single_num_privileged_obs = 73
        num_privileged_obs = int(c_frame_stack * (single_num_privileged_obs))
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/clpai_12dof_0905/urdf/clpai_12dof_0905_rl.urdf"
        name = "Pai"
        foot_name = "ankle_roll"
        knee_name = "calf"

        terminate_after_contacts_on = ["base_link"]
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        import math

        pi = math.pi
        degree_2_pi = 180.0 * pi
        pos = [0.0, 0.0, 0.3453]
        # rot = [0., 0.27154693695611287, 0., 0.962425197628238]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            "hip_pitch_joint": 80.0,
            "hip_roll_joint": 80.0,
            "thigh_joint": 80.0,
            "calf_joint": 80.0,
            "ankle_pitch_joint": 80,
            "ankle_roll_joint": 80,
        }
        damping = {
            "hip_pitch_joint": 0.25,
            "hip_roll_joint": 0.25,
            "thigh_joint": 0.25,
            "calf_joint": 0.25,
            "ankle_pitch_joint": 0.25,
            "ankle_roll_joint": 0.25,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz
        # decimation = 20  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 20
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.5]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.32
        min_dist_feet = 0.15 - 0.03
        max_dist_feet = 0.16 + 0.03

        min_dist_knee = 0.20 - 0.06
        max_dist_knee = 0.20 + 0.03
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.3  # rad
        target_feet_height = 0.03  # 0.052621126 # m
        cycle_time = 0.5  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 100  # forces above this value are penalized


        class scales:
            # reference motion tracking
            joint_pos = 1.6*1  # 1.6
            feet_clearance = 1.0
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.0
            foot_slip = -0.05
            feet_distance = 0.16  # 0.2
            knee_distance = 0.16  # 0.2
            # contact
            feet_contact_forces = -0.01
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.0
            
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.0
            base_height = 0.5
            base_acc = 0.2
            

            # termination = 1.
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0


class Pai_Cl_0905_CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 45  # per iteration
        max_iterations = 30001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "Pai_cl_0905_ppo"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt


# python scripts/train.py --task=pai_cl_0905_ppo --run_name v1 --headless --num_envs 4096
# python scripts/play.py --task=pai_cl_0905_ppo --run_name v1
# python scripts/sim2sim.py --load_model ../logs/Pai_cl_0905_ppo/exported/policies/policy_1.pt
