import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
import torch


def rt_mat(theta, d):
    """
    生成一个二维平面上的旋转和平移矩阵
    theta: 旋转角度（弧度）
    d: 平移向量 (dx, dy)
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    dx, dy, dz = d
    return torch.tensor(
        [
            [cos_theta, 0, -sin_theta, dx],
            [0, 1, 0, dy],
            [sin_theta, 0, cos_theta, dz],
            [0, 0, 0, 1],
        ]
    )


def plot_3D_xz_with_phase(ax, phase, x_toe, z_toe, _label):
    ax.plot(phase.numpy(), x_toe.numpy(), z_toe.numpy(), label=_label)
    ax.set_title("3D Toe Trajectory")
    ax.set_xlabel("Phase")
    ax.set_ylabel("X Position")
    ax.set_zlabel("Z Position")


def plot_2D(
    x,
    y,
    _label="Toe Trajectory",
    title="Toe Trajectory",
    xlable="Phase",
    ylabel="X Position",
):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=_label)
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylabel)
    plt.grid(True)
    # plt.show()


def leg(phase, sin_pos, cos_pos, side=1):
    scale_1 = 0.3  # 假设的 scale_1 值
    scale_2 = 2 * scale_1
    l0 = (0, side * 0.0233, -0.033)
    l1 = (0, 0, -0.1395)
    l2 = (0, 0, -0.14)
    l3 = (0.05, 0, 0)
    ref_dof_pos = torch.zeros((1000, 12))
    ref_dof_pos[:, 0] = sin_pos * scale_1
    ref_dof_pos[:, 3] = -sin_pos * scale_2
    ref_dof_pos[:, 4] = sin_pos * (scale_2 - scale_1)
    # ref_dof_pos[torch.abs(sin_pos) < 0.3] = 0
    ref_dof_pos[:, 0] = ref_dof_pos[:, 0] + 0.25
    ref_dof_pos[:, 3] = ref_dof_pos[:, 3] - 0.65
    ref_dof_pos[:, 4] = ref_dof_pos[:, 4] + 0.4

    rt_mat_0 = torch.stack([rt_mat(a, l0) for a in ref_dof_pos[:, 0]])
    rt_mat_1 = torch.stack([rt_mat(a, l1) for a in ref_dof_pos[:, 3]])
    rt_mat_2 = torch.stack([rt_mat(a, l2) for a in ref_dof_pos[:, 4]])
    rt_mat_3 = torch.stack([rt_mat(a, l3) for a in ref_dof_pos[:, 5]])
    total_rt_mat = rt_mat_0 @ rt_mat_1 @ rt_mat_2 @ rt_mat_3
    # print(total_rt_mat.size())
    foot_end_local = torch.tensor([0.0, 0.0, 0.0, 1.0])

    hip_pitch_base = (rt_mat_0 @ foot_end_local)[:, :3]
    knee_pitch_base = (rt_mat_0 @ rt_mat_1 @ foot_end_local)[:, :3]
    ankle_pitch_base = (rt_mat_0 @ rt_mat_1 @ rt_mat_2 @ foot_end_local)[:, :3]
    foot_end_base = (rt_mat_0 @ rt_mat_1 @ rt_mat_2 @ rt_mat_3 @ foot_end_local)[:, :3]

    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_ylim(-0.3, 0.3)
    # ax.set_zlim(-0.3, 0.3)
    # plot_3D_xz_with_phase(ax,phase,hip_pitch_base[:,0],hip_pitch_base[:,2],"hip_pitch")
    # plot_3D_xz_with_phase(ax,phase,knee_pitch_base[:,0],knee_pitch_base[:,2],"knee_pitch")
    # plot_3D_xz_with_phase(ax,phase,ankle_pitch_base[:,0],ankle_pitch_base[:,2],"ankle_pitch")
    # plot_3D_xz_with_phase(ax,phase,foot_end_base[:,0],foot_end_base[:,2],"foot_end")
    # ax.legend()
    # plt.figure(figsize=(10, 6))
    dist = torch.norm(knee_pitch_base - hip_pitch_base, dim=1).numpy()
    # print(dist)
    # plt.plot(phase.numpy(), dist, label="Toe Trajectory")
    # plt.title("Toe Trajectory")
    # plt.xlabel("Phase")
    # plt.ylabel("X Position")
    # plt.grid(True)
    # plt.show()
    return (hip_pitch_base, knee_pitch_base, ankle_pitch_base, foot_end_base)


class Leg:
    hip_pitch_base: torch.Tensor
    knee_pitch_base: torch.Tensor
    ankle_pitch_base: torch.Tensor
    foot_end_base: torch.Tensor

    def __init__(
        self,
        phase: torch.Tensor,
        ax,
        side="l",
        scale=0.3,
    ) -> None:
        self.phase: torch.Tensor = phase
        self.ax = ax
        # 定义参数
        self.T = 0.5  # 步态周期
        self.beta = 0.6  # 站姿相位的比例因子
        self.vx = 0.4  # x 方向速度
        self.vy = 0.0  # y 方向速度
        self.omega = 0.0  # 角速度
        self.lx = 1.0  # 腿长
        self.k_i = 1.0  # 常数
        self.h = 0.05

        self.l1 = (0, 0, -0.1395)
        self.l2 = (0, 0, -0.14)
        self.l3 = (0.05, 0, 0)
        self.ref_dof_pos = torch.zeros((self.phase.size(dim=0), 6))
        self.scale_1 = scale  # 假设的 scale_1 值
        self.scale_2 = 2 * self.scale_1
        self.init_pos()

        if side == "l":
            self.sin_pos = torch.sin(2 * torch.pi * self.phase)
            self.cos_pos = torch.cos(2 * torch.pi * self.phase)
            self.l0 = (0, 0.0233, -0.033)
            self.fresh_joint_angle_with_ref()

            # self.fresh_ref_joint_angle_with_sin()
            self.compute_r_i(0)
            self.fresh_ref_joint_angle_with_cycle()
            (self.line,) = ax.plot([], [], [], lw=2, label="left")
        elif side == "r":
            self.sin_pos = torch.sin(2 * torch.pi * self.phase + torch.pi)
            self.cos_pos = torch.cos(2 * torch.pi * self.phase + torch.pi)
            self.l0 = (0, -0.0233, -0.033)
            self.fresh_joint_angle_with_ref()

            # self.fresh_ref_joint_angle_with_sin()
            # self.compute_r_i(torch.pi)
            # self.fresh_ref_joint_angle_with_cycle()
            (self.line,) = ax.plot([], [], [], lw=2, label="right")
        print(self.phase.size())

        self.fresh_joint_angle_with_ref()

    def init_pos(self):
        self.ref_dof_pos[:, 0] = 0.25
        self.ref_dof_pos[:, 3] = -0.65
        self.ref_dof_pos[:, 4] = 0.4

    def fresh_joint_angle_with_ref(self):
        rt_mat_0 = torch.stack([rt_mat(a, self.l0) for a in self.ref_dof_pos[:, 0]])
        rt_mat_1 = torch.stack([rt_mat(a, self.l1) for a in self.ref_dof_pos[:, 3]])
        rt_mat_2 = torch.stack([rt_mat(a, self.l2) for a in self.ref_dof_pos[:, 4]])
        rt_mat_3 = torch.stack([rt_mat(a, self.l3) for a in self.ref_dof_pos[:, 5]])
        total_rt_mat = rt_mat_0 @ rt_mat_1 @ rt_mat_2 @ rt_mat_3
        # print(total_rt_mat.size())
        foot_end_local = torch.tensor([0.0, 0.0, 0.0, 1.0])

        self.hip_pitch_base = (rt_mat_0 @ foot_end_local)[:, :3]
        self.knee_pitch_base = (rt_mat_0 @ rt_mat_1 @ foot_end_local)[:, :3]
        self.ankle_pitch_base = (rt_mat_0 @ rt_mat_1 @ rt_mat_2 @ foot_end_local)[:, :3]
        self.foot_end_base = (
            rt_mat_0 @ rt_mat_1 @ rt_mat_2 @ rt_mat_3 @ foot_end_local
        )[:, :3]

    def fresh_ref_joint_angle_with_sin(self):
        self.ref_dof_pos[:, 0] += self.sin_pos * self.scale_1
        self.ref_dof_pos[:, 3] += -self.sin_pos * self.scale_2
        self.ref_dof_pos[:, 4] += self.sin_pos * (self.scale_2 - self.scale_1)

    def compute_p_i(self):
        phi_i = (self.phase % self.T) / self.T * 2 * torch.pi
        return phi_i

    def compute_r_i(self, start_phase=torch.pi):
        phyi = self.compute_p_i() + start_phase
        phyi[phyi > torch.pi] -= 2 * torch.pi
        pi = phyi.clone()
        pi[phyi < 0] = -phyi[phyi < 0] / torch.pi
        pi[phyi > 0] = phyi[phyi > 0] / torch.pi
        a_i_x = self.vx * self.T * self.beta
        a_i_y = (self.vy + self.k_i * self.omega * self.lx / 2) * self.T * self.beta

        add_x = a_i_x * (6 * pi**5 - 15 * pi**4 + 10 * pi**3 - 0.5)
        add_y = a_i_y * (6 * pi**5 - 15 * pi**4 + 10 * pi**3 - 0.5)
        add_z = self.h * (-64 * pi**6 + 192 * pi**5 - 192 * pi**4 + 64 * pi**3)
        self.r_i_x = self.ankle_pitch_base[:, 0] +add_x
        self.r_i_y = self.ankle_pitch_base[:, 1] +add_y
        self.r_i_z = self.ankle_pitch_base[:, 2] - self.hip_pitch_base[:, 2] +add_z
        self.r_i_z[phyi < 0] = (
            self.ankle_pitch_base[phyi < 0, 2] - self.hip_pitch_base[phyi < 0, 2]
        )
        # fig = plt.figure(figsize=(10, 6))
        # ax = fig.add_subplot(111, projection="3d")
        # plot_3D_xz_with_phase(ax, self.phase, self.r_i_x, self.r_i_z, "r_i_x r_i_z")

    def fresh_ref_joint_angle_with_cycle(self):
        print("fresh_ref_joint_angle_with_cycle")
        print(self.r_i_x.size())
        l1 = self.l1[2]
        l2 = self.l2[2]

        d = (self.r_i_x**2 + self.r_i_z**2 - l1**2 - l2**2) / (2 * l1 * l2)
        sita2 = torch.atan2(torch.sqrt(1 - d**2), d)
        sita2[sita2<-torch.pi]+=2*torch.pi
        print(sita2[:5])
        plot_2D(self.phase, -sita2, ylabel="sita calf", _label="angle 2")

        sita1 = torch.atan2(
            l2 * torch.sin(sita2), l1 + l2 * torch.cos(sita2)
        ) - torch.atan2(self.r_i_x, self.r_i_z)
        sita1[sita1<-torch.pi]+=2*torch.pi
        
        plot_2D(self.phase, sita1, ylabel="sita thigh", _label="angle 1")
        self.ref_dof_pos[:, 0] = sita1
        self.ref_dof_pos[:, 3] = -sita2
        self.ref_dof_pos[:, 4] = sita2 - sita1

    def init_line(self):
        self.line.set_data([], [])
        self.line.set_3d_properties([])

    def update_lin(self, num):
        x = [
            self.hip_pitch_base[num, 0],
            self.knee_pitch_base[num, 0],
            self.ankle_pitch_base[num, 0],
            self.foot_end_base[num, 0],
        ]
        z = [
            self.hip_pitch_base[num, 2],
            self.knee_pitch_base[num, 2],
            self.ankle_pitch_base[num, 2],
            self.foot_end_base[num, 2],
        ]
        print(z)
        print(phase[num])
        self.line.set_data([phase[num],phase[num],phase[num],phase[num]], z)
        self.line.set_3d_properties(x)


class plot_two_leg:
    def __init__(self, phase: torch.Tensor) -> None:
        self.phase: torch.Tensor = phase.clone()
        print("1")
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.l_leg = Leg(phase=self.phase, ax=self.ax, side="l")
        self.r_leg = Leg(phase=self.phase, ax=self.ax, side="r")
        self.set_ax()

    def set_ax(self):
        # 设置轴的范围
        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(-0.35, 0.05)
        self.ax.set_zlim(-0.2, 0.2)

        # 设置标签
        self.ax.set_xlabel("Y Label") # y
        self.ax.set_ylabel("Z Label")
        self.ax.set_zlabel("X Label") # x

        ani = FuncAnimation(
            self.fig,
            self.update_line,
            frames=len(self.phase),
            init_func=self.init_line,
            blit=True,
            interval=50,
        )

        # 显示图形
        plt.show()

    def init_line(self):
        self.l_leg.init_line()
        self.r_leg.init_line()
        return (self.l_leg.line, self.r_leg.line)

    def update_line(self, num):
        self.l_leg.update_lin(num)
        self.r_leg.update_lin(num)
        return (self.l_leg.line, self.r_leg.line)


if __name__ == "__main__":

    phase = torch.linspace(0, 4, 1000)
    aa = plot_two_leg(phase)
