import numpy as np
from ikpy.chain import Chain
from ikpy.link import Link
from scipy.spatial.transform import Rotation as R
from typing import List
import matplotlib.pyplot as plt

urdf = "/home/hpx/HPXLoco/livelybot_rl_baseline-main/resources/robots/clpai_12dof_0905/urdf/clpai_12dof_0905_rl.urdf"
# 加载 URDF 文件并初始化链
# 加载 URDF 文件并初始化链
left_leg_chain: Chain = Chain.from_urdf_file(
    urdf,
    base_elements=["base_link"],
    last_link_vector=[0, 0, 0],
    base_element_type="link",
)
right_leg_chain: Chain = Chain.from_urdf_file(
    urdf,
    base_elements=["base_link"],
    last_link_vector=[0, 0, 0],
    base_element_type="link",
)

# 可视化
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# 打印链条信息以检查链接数量
print("左腿链条链接数量:", len(left_leg_chain.links))
links: List[Link] = left_leg_chain.links
for l in links:
    print(l.name)
print("右腿链条链接数量:", len(right_leg_chain.links))

# 确定每条腿的活动关节掩码
left_leg_active_links_mask = [False, True, True, False, True, True, True, False]
right_leg_active_links_mask = [False, True, True, False, True, True, True, False]

# 更新链的活动关节掩码
left_leg_chain.active_links_mask = left_leg_active_links_mask
right_leg_chain.active_links_mask = right_leg_active_links_mask

# 当前关节角度
current_joint_angles_left = [0, 0, 0, 0, 0, 0, 0, 0]
current_joint_angles_right = [0, 0, 0, 0, 0, 0, 0, 0]
# 使用正运动学求解足端的位置与姿态
end_effector_frame_left = left_leg_chain.forward_kinematics(current_joint_angles_left)
end_effector_frame_right = right_leg_chain.forward_kinematics(
    current_joint_angles_right
)
left_leg_chain.plot(current_joint_angles_left, ax, target=end_effector_frame_left, show=False)

# 提取位置和姿态
current_position_left = end_effector_frame_left[:3, 3]
current_orientation_left = end_effector_frame_left[:3, :3]
print(current_position_left,"\r\n",current_orientation_left)
current_position_right = end_effector_frame_right[:3, 3]
current_orientation_right = end_effector_frame_right[:3, :3]
print(current_position_right,"\r\n",current_orientation_right)

# 目标位置和姿态（欧拉角表示）
target_position_left = current_position_left
current_position_left[2] += 0.02
# target_orientation_left_euler = [roll1, pitch1, yaw1]  # 欧拉角

target_position_right = current_position_right
current_position_right[2] += 0.05
# target_orientation_right_euler = [roll2, pitch2, yaw2]  # 欧拉角

# 将欧拉角转换为旋转矩阵
# rotation_matrix_left = R.from_euler("xyz", target_orientation_left_euler).as_matrix()
# rotation_matrix_right = R.from_euler("xyz", target_orientation_right_euler).as_matrix()

# 创建目标变换矩阵
target_frame_left = np.eye(4)
target_frame_left[:3, :3] = current_orientation_left
target_frame_left[:3, 3] = current_position_left

target_frame_right = np.eye(4)
target_frame_right[:3, :3] = current_orientation_right
target_frame_right[:3, 3] = current_position_right

# 使用 ikpy 进行逆运动学求解
joint_angles_solution_left = left_leg_chain.inverse_kinematics_frame(
    target_frame_left, initial_position=current_joint_angles_left
)
joint_angles_solution_right = right_leg_chain.inverse_kinematics_frame(
    target_frame_right, initial_position=current_joint_angles_right
)

print("左腿关节角度解:", joint_angles_solution_left)
print("右腿关节角度解:", joint_angles_solution_right)


# 绘制右腿
# right_leg_chain.plot(joint_angles_solution_right, ax, target=target_frame_right, show=False)
left_leg_chain.plot(joint_angles_solution_left, ax, target=target_frame_left, show=False)

# 设置绘图参数
ax.set_xlim(-0.3, 0.1)
ax.set_ylim(-0.1, 0.3)
ax.set_zlim(-0.3, 0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Biped Robot Legs')

# 显示绘图
plt.show()