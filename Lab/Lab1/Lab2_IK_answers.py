import numpy as np
from scipy.spatial.transform import Rotation as R

def CCD(meta_data, joint_positions, joint_orientations, target_pose):
    # 计算 inverse_kinematics 链的信息
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    path_positions = []
    for joint in path:
        path_positions.append(joint_positions[joint])
    path_offsets = []
    path_offsets.append(np.array([0., 0., 0.]))
    for i in range(len(path) - 1):
        path_offsets.append(meta_data.joint_initial_position[path[i + 1]] - meta_data.joint_initial_position[path[i]])
    path_orientations = []
    for i in range(len(path2) - 1):
        path_orientations.append(R.from_quat(joint_orientations[path2[i + 1]]))
    path_orientations.append(R.from_quat(joint_orientations[path2[-1]]))
    for i in range(len(path1) - 1):
        path_orientations.append(R.from_quat(joint_orientations[path1[~i]]))
    path_orientations.append(R.identity())
    
    # CCD 循环
    cnt = 0
    end_index = path_name.index(meta_data.end_joint)
    while (np.linalg.norm(joint_positions[path[end_index]] - target_pose) >= 1e-2 and cnt <= 10):
        for i in range(end_index):
            current_index = end_index - i - 1
            current_position = path_positions[current_index]
            end_position = path_positions[end_index]
            vector_current2end = end_position - current_position
            vector_current2target = target_pose - current_position
            current2end = vector_current2end / np.linalg.norm(vector_current2end)
            current2target = vector_current2target / np.linalg.norm(vector_current2target)

            # 计算轴角
            rotation_radius = np.arccos(np.clip(np.dot(current2end, current2target), -1, 1))
            temp_axis = np.cross(current2end, current2target)
            rotation_axis = temp_axis / np.linalg.norm(temp_axis)
            rotation_vector = R.from_rotvec(rotation_radius * rotation_axis)
            
            # 计算方位与位置
            path_orientations[current_index] = rotation_vector * path_orientations[current_index]
            path_rotations = []
            path_rotations.append(path_orientations[0])
            for j in range(len(path_orientations) - 1):
                path_rotations.append(R.inv(path_orientations[j]) * path_orientations[j + 1])
            for j in range(current_index, end_index):
                path_positions[j + 1] = path_positions[j] + path_orientations[j].apply(path_offsets[j + 1])
                if j + 1 < end_index:
                    path_orientations[j + 1] = path_orientations[j] * path_rotations[j + 1]
                else:
                    path_orientations[j + 1] = path_orientations[j]
        cnt += 1

    return path_positions, path_orientations

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path_positions, path_orientations = CCD(meta_data, joint_positions, joint_orientations, target_pose)

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    # 计算 path_joints 的旋转
    joint_rotations = R.identity(len(meta_data.joint_name))
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            joint_rotations[i] = R.from_quat(joint_orientations[i])
        else:
            joint_rotations[i] = R.inv(R.from_quat(joint_orientations[meta_data.joint_parent[i]])) * R.from_quat(joint_orientations[i])
    
    # path_joints 的 forwar_kinematics
    for i in range(len(path2) - 1):
        joint_orientations[path2[i + 1]] = path_orientations[i].as_quat()
    joint_orientations[path2[-1]] = path_orientations[len(path2) - 1].as_quat()
    for i in range(len(path1) - 1):
        joint_orientations[path1[~i]] = path_orientations[i + len(path2)].as_quat()

    for i in range(len(path)):
        joint_positions[path[i]] = path_positions[i]

    # 其余 joints 的 forward_kinematics
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            continue
        if meta_data.joint_name[i] not in path_name:
            joint_positions[i] = joint_positions[meta_data.joint_parent[i]] + \
                R.from_quat(joint_orientations[meta_data.joint_parent[i]]).apply(meta_data.joint_initial_position[i] - \
                meta_data.joint_initial_position[meta_data.joint_parent[i]])
            joint_orientations[i] = (R.from_quat(joint_orientations[meta_data.joint_parent[i]]) * joint_rotations[i]).as_quat()
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pose = np.array([joint_positions[0][0] + relative_x, target_height, joint_positions[0][2] + relative_z])
    path_positions, path_orientations = CCD(meta_data, joint_positions, joint_orientations, target_pose)
    
    path, path_name, _, _ = meta_data.get_path_from_root_to_end()

    # 计算 path_joints 的旋转
    joint_rotations = R.identity(len(meta_data.joint_name))
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            joint_rotations[i] = R.from_quat(joint_orientations[i])
        else:
            joint_rotations[i] = R.inv(R.from_quat(joint_orientations[meta_data.joint_parent[i]])) * R.from_quat(joint_orientations[i])

    # path_joints 的 forwar_kinematics
    for j in range(len(path)):
        joint_positions[path[j]] = path_positions[j]
        joint_orientations[path[j]] = path_orientations[j].as_quat()

    # 其余 joints 的 forward_kinematics
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            continue
        if meta_data.joint_name[i] not in path_name:
            joint_positions[i] = joint_positions[meta_data.joint_parent[i]] + \
                R.from_quat(joint_orientations[meta_data.joint_parent[i]]).apply(meta_data.joint_initial_position[i] - \
                meta_data.joint_initial_position[meta_data.joint_parent[i]])
            joint_orientations[i] = (R.from_quat(joint_orientations[meta_data.joint_parent[i]]) * joint_rotations[i]).as_quat()
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations