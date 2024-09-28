import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation_matrix(angle, z, x):
    """
    生成一个二维旋转矩阵
    :param angle: 旋转角度（弧度）
    :return: 2x2 旋转矩阵
    """
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    return torch.tensor(
        [
            [
                cos_a,
                -sin_a,
            ],
            [sin_a, cos_a],
        ]
    )


def rotation_translation_matrix(angle, translation):
    """
    生成一个二维旋转平移矩阵
    :param angle: 旋转角度（弧度）
    :param translation: 平移向量（x, y）
    :return: 3x3 旋转平移矩阵
    """
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    return torch.tensor(
        [[cos_a, -sin_a, translation[0]], [sin_a, cos_a, translation[1]], [0, 0, 1]]
    )


def plot_3D_xz(ax, phase, total_rt_mat, _label):
    x_toe_l = total_rt_mat[:, 0, 2]
    z_toe_l = total_rt_mat[:, 1, 2]
    pit = torch.atan2(total_rt_mat[:, 1, 0], total_rt_mat[:, 0, 0])
    ax.plot(phase.numpy(), x_toe_l.numpy(), z_toe_l.numpy(), label=_label)
    ax.set_title("3D Toe Trajectory")
    ax.set_xlabel("Phase")
    ax.set_ylabel("X Position")
    ax.set_zlabel("Z Position")


def plot_3D_leg(phase, rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_3D_xz(
        ax,
        phase,
        torch.stack(
            [
                rt0
                for rt0, rt1, rt2, rt3 in zip(
                    rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l
                )
            ]
        ),
        "1",
    )
    plot_3D_xz(
        ax,
        phase,
        torch.stack(
            [
                rt0 @ rt1
                for rt0, rt1, rt2, rt3 in zip(
                    rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l
                )
            ]
        ),
        "2",
    )
    plot_3D_xz(
        ax,
        phase,
        torch.stack(
            [
                rt0 @ rt1 @ rt2
                for rt0, rt1, rt2, rt3 in zip(
                    rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l
                )
            ]
        ),
        "3",
    )
    plot_3D_xz(
        ax,
        phase,
        torch.stack(
            [
                rt0 @ rt1 @ rt2 @ rt3
                for rt0, rt1, rt2, rt3 in zip(
                    rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l
                )
            ]
        ),
        "4",
    )
    ax.legend()
    plt.show()


# 假设 phase 是一个从 0 到 1 的张量
phase = torch.linspace(0, 4, 1000)

# 计算 sin_pos, sin_pos_l 和 sin_pos_r
sin_pos_l = torch.sin(2 * torch.pi * phase)
# sin_pos_l = sin_pos.clone()
sin_pos_r = torch.sin(2 * torch.pi * phase + torch.pi)
# sin_pos_r = sin_pos.clone()

# 设置参数
scale_0 = 0.2  # 假设的 scale_1 值
scale_1 = 0.2  # 假设的 scale_1 值
scale_2 = 2 * scale_1
l1 = 0.1395
l2 = 0.14
l3 = 0.03

# left foot stance phase set to default joint pos
# sin_pos_l[sin_pos_l > 0] = 0
ref_dof_pos_l = torch.zeros((1000, 12))
ref_dof_pos_l[:, 0] = sin_pos_l * scale_1
ref_dof_pos_l[:, 3] = -sin_pos_l * scale_2
ref_dof_pos_l[:, 4] = sin_pos_l * scale_1
ref_dof_pos_l[torch.abs(sin_pos_l) < 0.1] = 0
ref_dof_pos_l[:, 0] = ref_dof_pos_l[:, 0] + 0.25
ref_dof_pos_l[:, 3] = ref_dof_pos_l[:, 3] - 0.65
ref_dof_pos_l[:, 4] = ref_dof_pos_l[:, 4] + 0.4
# -0.25 0.65 -0.4
rt_mat_0_l = torch.stack(
    [rotation_translation_matrix(angle1, (0, 0)) for angle1 in ref_dof_pos_l[:, 0]]
)
rt_mat_1_l = torch.stack(
    [rotation_translation_matrix(angle1, (0, -l1)) for angle1 in ref_dof_pos_l[:, 3]]
)
rt_mat_2_l = torch.stack(
    [rotation_translation_matrix(angle1, (0, -l2)) for angle1 in ref_dof_pos_l[:, 4]]
)
rt_mat_3_l = torch.stack(
    [
        rotation_translation_matrix(angle1-angle1, (l3, 0))
        for angle1 in ref_dof_pos_l[:, 4]
    ]
)
total_rt_mat_l_3 = torch.stack([rt0 @ rt1 @ rt2 @ rt3 for rt0, rt1, rt2, rt3 in zip(rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l)])
total_rt_mat_l_2 = torch.stack([rt0 @ rt1 @ rt2 for rt0, rt1, rt2, rt3 in zip(rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l)])
total_rt_mat_l_1 = torch.stack([rt0 @ rt1 for rt0, rt1, rt2, rt3 in zip(rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l)])
total_rt_mat_l_0 = torch.stack([rt0 for rt0, rt1, rt2, rt3 in zip(rt_mat_0_l, rt_mat_1_l, rt_mat_2_l, rt_mat_3_l)])
rotation_matrices = total_rt_mat_l_3[:, :2, :2]

# print(total_rt_mat[:,:2,2])
x_toe_l = total_rt_mat_l_3[:, 0, 2]
z_toe_l = total_rt_mat_l_3[:, 1, 2]
pit_rot = torch.atan2(total_rt_mat_l_3[:, 1, 0], total_rt_mat_l_3[:, 0, 0]) / torch.pi * 180
print(z_toe_l.max(), z_toe_l.min(), "z dist:", z_toe_l.max() - z_toe_l.min())
print(x_toe_l.max(), x_toe_l.min(), "x dist:", x_toe_l.max() - x_toe_l.min())

# plot_3D_leg(phase, rt_mat_0_l.inverse(), rt_mat_1_l.inverse(), rt_mat_2_l.inverse(), rt_mat_3_l.inverse())

# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# plt.plot(phase.numpy(), x_toe_l.numpy(), label="Toe Trajectory")
# plt.title("Toe Trajectory")
# plt.xlabel("Phase")
# plt.ylabel("X Position")
# plt.grid(True)
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(phase.numpy(), z_toe_l.numpy(), label="Toe Trajectory")
# plt.title("Toe Trajectory")
# plt.xlabel("Phase")
# plt.ylabel("Z Position")
# plt.grid(True)
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(phase.numpy(), pit_rot.numpy(), label="Toe Trajectory")
# plt.title("Toe Trajectory")
# plt.xlabel("Phase")
# plt.ylabel("Pitch")
# plt.grid(True)
# plt.legend()
# plt.show()

# right foot stance phase set to default joint pos
# sin_pos_r[sin_pos_r < 0] = 0
ref_dof_pos_r = torch.zeros((1000, 12))  # 假设有 11 个关节
ref_dof_pos_r[:, 6] = sin_pos_r * scale_1 + 0.25
ref_dof_pos_r[:, 9] = -sin_pos_r * scale_2 - 0.65
ref_dof_pos_r[:, 10] = sin_pos_r * scale_1 + 0.4

rt_mat_0_r = torch.stack(
    [rotation_translation_matrix(angle1, (0, 0)) for angle1 in ref_dof_pos_r[:, 6]]
)
rt_mat_1_r = torch.stack(
    [rotation_translation_matrix(angle1, (0, -l1)) for angle1 in ref_dof_pos_r[:, 9]]
)
rt_mat_2_r = torch.stack(
    [rotation_translation_matrix(angle1, (0, -l2)) for angle1 in ref_dof_pos_r[:, 10]]
)
rt_mat_3_r = torch.stack(
    [
        rotation_translation_matrix(angle1 - angle1, (l3, 0))
        for angle1 in ref_dof_pos_r[:, 10]
    ]
)
total_rt_mat = torch.stack(
    [
        rt0 @ rt1 @ rt2 @ rt3
        for rt0, rt1, rt2, rt3 in zip(rt_mat_0_r, rt_mat_1_r, rt_mat_2_r, rt_mat_3_r)
    ]
)
total_rt_mat_r_3 = torch.stack([rt0 @ rt1 @ rt2 @ rt3 for rt0, rt1, rt2, rt3 in zip(rt_mat_0_r, rt_mat_1_r, rt_mat_2_r, rt_mat_3_r)])
total_rt_mat_r_2 = torch.stack([rt0 @ rt1 @ rt2 for rt0, rt1, rt2, rt3 in zip(rt_mat_0_r, rt_mat_1_r, rt_mat_2_r, rt_mat_3_r)])
total_rt_mat_r_1 = torch.stack([rt0 @ rt1 for rt0, rt1, rt2, rt3 in zip(rt_mat_0_r, rt_mat_1_r, rt_mat_2_r, rt_mat_3_r)])
total_rt_mat_r_0 = torch.stack([rt0 for rt0, rt1, rt2, rt3 in zip(rt_mat_0_r, rt_mat_1_r, rt_mat_2_r, rt_mat_3_r)])

rotation_matrices = total_rt_mat[:, :2, :2]
x_toe_l = total_rt_mat[:, 0, 2]
z_toe_l = total_rt_mat[:, 1, 2]

# plot_3D_leg(phase, rt_mat_0_r, rt_mat_1_r, rt_mat_2_r, rt_mat_3_r)

# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(phase.numpy(), x_toe_l.numpy(), label="Toe Trajectory")
# plt.title("Toe Trajectory")
# plt.xlabel("Phase")
# plt.ylabel("X Position")
# plt.grid(True)
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(phase.numpy(), z_toe_l.numpy(), label="Toe Trajectory")
# plt.title("Toe Trajectory")
# plt.xlabel("Phase")
# plt.ylabel("Z Position")
# plt.grid(True)
# plt.legend()
# plt.show()


# 绘制图形
# plt.figure(figsize=(12, 8))

# # 绘制 sin_pos_l
# plt.subplot(2, 2, 1)
# plt.plot(phase.numpy(), sin_pos_l.numpy(), label="sin_pos_l")
# plt.title("sin_pos_l")
# plt.xlabel("Phase")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)

# # 绘制 sin_pos_r
# plt.subplot(2, 2, 2)
# plt.plot(phase.numpy(), sin_pos_r.numpy(), label="sin_pos_r")
# plt.title("sin_pos_r")
# plt.xlabel("Phase")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)

# # 绘制 ref_dof_pos
# plt.subplot(2, 2, 3)
# # for i in range(ref_dof_pos_l.shape[1]):
# plt.plot(phase.numpy(), ref_dof_pos_l[:, 0].numpy(), label=f"ref_dof_pos_l[:, {0}]")
# plt.plot(phase.numpy(), ref_dof_pos_l[:, 3].numpy(), label=f"ref_dof_pos_l[:, {3}]")
# plt.plot(phase.numpy(), ref_dof_pos_l[:, 4].numpy(), label=f"ref_dof_pos_l[:, {4}]")
# plt.title("ref_dof_pos")
# plt.xlabel("Phase")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(
#     phase.numpy(),
#     ref_dof_pos_r[:, 6].numpy(),
#     label=f"ref_dof_pos_r[:, {6}]",
#     linestyle="--",
# )
# plt.plot(
#     phase.numpy(),
#     ref_dof_pos_r[:, 9].numpy(),
#     label=f"ref_dof_pos_r[:, {9}]",
#     linestyle="--",
# )
# plt.plot(
#     phase.numpy(),
#     ref_dof_pos_r[:, 10].numpy(),
#     label=f"ref_dof_pos_r[:, {10}]",
#     linestyle="--",
# )

# plt.title("ref_dof_pos")
# plt.xlabel("Phase")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
