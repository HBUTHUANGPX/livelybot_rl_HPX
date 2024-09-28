import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


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
def plot_3D_xz_with_phase(ax, phase, x_toe, z_toe, _label):
    ax.plot(phase.numpy(), x_toe.numpy(), z_toe.numpy(), label=_label)
    ax.set_title("3D Toe Trajectory")
    ax.set_xlabel("Phase")
    ax.set_ylabel("X Position")
    ax.set_zlabel("Z Position")

# 定义参数
T = 0.5  # 步态周期
beta = 0.6  # 站姿相位的比例因子
vx = -0.4  # x 方向速度
vy = 0.0  # y 方向速度
omega = 0.0  # 角速度
lx = 1.0  # 腿长
k_i = 1.0  # 常数
h = 0.05
# 定义时间向量
phase = torch.linspace(0, 4, 1000)


# 计算 p_i
def compute_p_i(phase):
    phi_i = (phase % T) / T * 2 * torch.pi
    return phi_i

# 计算 r_i_x, r_i_y, r_i_z
def compute_r_i(phase, start_phase=torch.pi):
    phyi = compute_p_i(phase) + start_phase
    phyi[phyi > torch.pi] -= 2 * torch.pi
    pi = phyi.clone()
    pi[phyi < 0] =   -phyi[phyi < 0] / torch.pi
    pi[phyi > 0] =   phyi[phyi > 0] / torch.pi
    a_i_x = vx * T * beta
    a_i_y = (vy + k_i * omega * lx / 2) * T * beta

    r_i_x = a_i_x * (6 * pi**5 - 15 * pi**4 + 10 * pi**3 - 0.5)
    r_i_y = a_i_y * (6 * pi**5 - 15 * pi**4 + 10 * pi**3 - 0.5)

    stance = (phase % T < T / 2).float()
    swing = (phase % T >= T / 2).float()
    r_i_z = h * (-64 * pi**6 + 192 * pi**5 - 192 * pi**4 + 64 * pi**3)
    r_i_z[phyi < 0] = 0
    return r_i_x, r_i_y, r_i_z, pi, phyi

# 计算曲线数据
r_i_x, r_i_y, r_i_z, pi, phyi = compute_r_i(phase, start_phase=0)
# plot_2D(phase,pi)
# r_i_x, r_i_y, r_i_z ,pi, phyi= compute_r_i(phase,start_phase = torch.pi)
# plot_2D(phase,phase)
# plot_2D(phase, (phyi),ylabel="phy", title="[-pi,pi]")
# plot_2D(phase, pi, ylabel="pi",title="[-1,1]")
# plot_2D(phase,r_i_x,ylabel="X",title="X")
# plot_2D(phase,r_i_y,ylabel="Y")
# plot_2D(phase, r_i_z, ylabel="Z",title="Z")
# plot_2D(r_i_x, r_i_z, ylabel="z",title="XZ")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
plot_3D_xz_with_phase(ax,phase,r_i_x,r_i_z,"foot_end")



plt.show()
