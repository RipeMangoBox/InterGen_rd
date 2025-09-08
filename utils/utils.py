import os
import numpy as np
import torch
# import cv2
from PIL import Image
import math
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from common.quaternion import *

from utils.rotation_conversions import *

face_joint_indx = [2,1,17,16]
fid_l = [7,10]
fid_r = [8,11]


def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1,2,4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21])-1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20])-1
    left_hand_chain = np.array([22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30,])-1
    right_hand_chain = np.array([43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,])-1

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right(data, n_joints):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    return new_data


def rigid_transform(relative, data):

    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    global_vel = data[..., 22 * 3:22 * 6].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)
    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    global_positions[..., [0, 2]] += relative_t
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))
    global_vel = qrot_np(qinv_np(relative_r_rot_quat), global_vel)
    data[..., 22 * 3:22 * 6] = global_vel.reshape(data.shape[:-1] + (-1,))

    return data

def rigid_transform_dd100lf(relative, data):
    # 现在z轴为身高方向，旋转轴应为z轴
    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    global_vel = data[..., 22 * 3:22 * 6].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]
    # 构造绕z轴的四元数
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    # 配套一，getitem中angle = np.arctan2(r_relative[:, 3:4], r_relative[:, 0:1])
    # relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    # relative_r_rot_quat[..., 2] = np.sin(relative_rot)
    
    # 配套二：getitem中angle = 2 * np.arctan2(r_relative[:, 3:4], r_relative[:, 0:1])
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 3] = np.sin(relative_rot)  # z分量

    # 旋转
    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    # 平移
    global_positions[..., [0, 1]] += relative_t  # x, y平移
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))
    global_vel = qrot_np(qinv_np(relative_r_rot_quat), global_vel)
    data[..., 22 * 3:22 * 6] = global_vel.reshape(data.shape[:-1] + (-1,))

    return data

class MotionNormalizer():
    def __init__(self):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")

        self.motion_mean = mean
        self.motion_std = std


    def forward(self, x):
        x = (x - self.motion_mean) / self.motion_std
        return x

    def backward(self, x):
        x = x * self.motion_std + self.motion_mean
        return x



# class MotionNormalizerTorch():
#     def __init__(self):
#         mean = np.load("./data/global_mean.npy")
#         std = np.load("./data/global_std.npy")

#         self.motion_mean = torch.from_numpy(mean).float()
#         self.motion_std = torch.from_numpy(std).float()


#     def forward(self, x):
#         device = x.device
#         x = x.clone()
#         x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
#         return x

#     def backward(self, x, global_rt=False):
#         device = x.device
#         x = x.clone()
#         x = x * self.motion_std.to(device) + self.motion_mean.to(device)
#         return x

class MotionNormalizerTorch():

    """
    Torch version of MotionNormalizerRD.
    """
    def __init__(self, mode='both'):
        try:
            if mode == 'both':
                mean = np.load(f"./data/normalizer/rd/global_mean_rd.npy")
                std = np.load(f"./data/normalizer/rd/global_std_rd.npy")
            elif mode == 'leader':
                mean = np.load(f"./data/normalizer/rd/global_mean_rdl.npy")
                std = np.load(f"./data/normalizer/rd/global_std_rdl.npy")
            elif mode == 'follower':
                mean = np.load(f"./data/normalizer/rd/global_mean_rdf.npy")
                std = np.load(f"./data/normalizer/rd/global_std_rdf.npy")
              
            self.motion_mean = torch.from_numpy(mean).float()
            self.motion_std = torch.from_numpy(std).float()
            
            eps = 1e-8
            # 防止除以0，实际针对left_foot和right_foot joint的rot，即motion representation 的 (tensor([186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]),)
            zero_mask = abs(self.motion_std) < eps
            self.motion_std = torch.where(zero_mask, 1, self.motion_std)
            self.motion_mean = torch.where(zero_mask, 0, self.motion_mean)
        except:
            print(f"Unknown mode: {mode} or File not found or not loaded")

    def forward(self, x):
        device = x.device
        return (x - self.motion_mean.to(device)) / self.motion_std.to(device)

    def backward(self, x):
        device = x.device
        return x * self.motion_std.to(device) + self.motion_mean.to(device)


class MusicNormalizerRDTorch():
    """
    Torch version of MusicNormalizerRD.
    """
    def __init__(self):
        try:
            mean = np.load("./data/normalizer/rd/global_mean_rd_music.npy")
            std = np.load("./data/normalizer/rd/global_std_rd_music.npy")
            self.music_mean = torch.from_numpy(mean).float()
            self.music_std = torch.from_numpy(std).float()
        except:
            print("File not found or not loaded")

    def forward(self, x):
        device = x.device
        return (x - self.music_mean.to(device)) / self.music_std.to(device)

    def backward(self, x):
        device = x.device
        return x * self.music_std.to(device) + self.music_mean.to(device)
    
trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def process_motion_np(motion, feet_thre, prev_frames, n_joints):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)
    rotations = motion[:, n_joints*3:]

    positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height


    '''XZ at origin'''
    root_pos_init = positions[prev_frames]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init


    positions = qrot_np(root_quat_init_for_all, positions)

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)


    '''Get Joint Rotation Representation'''
    rot_data = rotations

    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1]
    data = np.concatenate([data, joint_vels], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, root_quat_init, root_pose_init_xz[None]





def process_dd100lf_motion_np(motion, feet_thre, prev_frames, n_joints):
    # (seq_len, joints_num, 3), motion=(seq_len, joints_num*3+rot_num)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]
    up_axis = 2 # z up (now z is height)
    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)
    rotations = motion[:, n_joints*3:]

    # 不再变换坐标轴，假设输入已经是xy平面，z为高度
    # positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)  # 不再需要

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[2]  # z为高度
    positions[:, :, 2] -= floor_height

    '''XY at origin'''
    root_pos_init = positions[prev_frames]
    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])  # 只保留xy，z为0
    positions = positions - root_pose_init_xy

    '''All initially face Y+ (正y轴)'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around z-axis
    forward_init = np.cross(across, np.array([[0, 0, 1]]), axis=-1)  # z轴朝上，右手定则
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 1, 0]])  # y轴正方向
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init_for_all, positions)

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 2]  # z为高度
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 2]  # z为高度
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Get Joint Rotation Representation'''
    rot_data = rotations

    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1] # [frame, 66=(joints)22*3] 0+66=66
    data = np.concatenate([data, joint_vels], axis=-1) # [frame, 66=(joints)22*6], 66+66=132
    data = np.concatenate([data, rot_data[:-1]], axis=-1) # [frame, 126=21*6], 132+126=258
    data = np.concatenate([data, feet_l, feet_r], axis=-1) # both feet are [frame, 2], 258+2+2=262

    return data, root_quat_init, root_pose_init_xy[None]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

MISSING_VALUE = -1

def save_image(image_numpy, image_path):
    img_pil = Image.fromarray(image_numpy)
    img_pil.save(image_path)


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, lr=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None and lr is not None :
        print('epoch: %3d niter:%6d inner_iter:%4d lr:%5f' % (epoch, niter_state, inner_iter, lr), end=" ")
    elif epoch is not None:
        print('epoch: %3d niter:%6d inner_iter:%4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


def compose_gif_img_list(img_list, fp_out, duration):
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)


def save_images(visuals, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = '%d_%s.jpg' % (i, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def save_images_test(visuals, image_path, from_name, to_name):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = "%s_%s_%s" % (from_name, to_name, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    # print(col, row)
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    # print(img_path)
    compose_img.save(img_path)


def compose_image(img_list, col, row, img_size):
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            # print((x * img_size[0], y*img_size[1],
            #                           (x + 1) * img_size[0], (y + 1) * img_size[1]))
            paste_area = (x * img_size[0], y*img_size[1],
                                      (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
            # to_image[y*img_size[1]:(y + 1) * img_size[1], x * img_size[0] :(x + 1) * img_size[0]] = from_img
    return to_image


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    # print(motion.shape)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

def save_pos3d(posf, posl, evaldir, fname, suffix='pos3d_npy'):
    save_folder = os.path.join(evaldir, suffix)
    os.makedirs(save_folder, exist_ok=True)
    
    np.save(os.path.join(save_folder, fname + '_00'), posf)
    np.save(os.path.join(save_folder, fname + '_01'), posl)